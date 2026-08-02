"""Fase 4a (PR7) — continuous multi-object tracking.

A pure-numpy SORT-style Kalman filter (`KalmanBoxTracker`) plus a
ByteTrack-style two-stage association tracker (`ByteTracker`), and the
`assign_track_ids()` driver that runs the latter as a sequential pass over
one video/batch's frames.

This is the first genuinely new algorithmic capability the security overhaul
adds (every prior PR was a refactor, an inert addition, or a runtime swap).
It plugs into `orchestrator.multi_models()` as an alternate label source:
when `security_config.SECURITY_RULES["label_source"] == "tracker"`,
`assign_track_ids()` replaces `YOLOPersonReID.cluster_locally()` as the
producer of the `labels` array `enroll_and_assign_global_ids()` consumes.
`enroll_and_assign_global_ids()` itself is untouched — it is agnostic to
where `labels` came from, which is what keeps this addition low-risk.

Why ByteTrack's two-stage idea matters here specifically: this pipeline's
person crops come from a security camera feed, where partial occlusion
(someone walking behind furniture, another person, etc.) is common. A
tracker that discards every low-confidence detection loses the track the
moment confidence dips, then re-identifies the person as a "new" track a
few frames later — exactly the kind of churn a security system cannot
afford (it would generate spurious enter/leave events). Recovering occluded
tracks via low-confidence detections, instead of just extrapolating blind
through `max_age` frames, gets the correct box back under the track the
moment it is visible again, even faintly.
"""

from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from camerachatbot.geometry.bbox_utils import iou

if TYPE_CHECKING:  # pragma: no cover - import-cycle avoidance only
    from camerachatbot.detectors.person_reid import YOLOPersonReID


def _bbox_to_z(bbox) -> Tuple[float, float, float, float]:
    """[x1, y1, x2, y2] -> (cx, cy, s, r) — center, area (scale), aspect ratio."""
    x1, y1, x2, y2 = bbox
    w = float(x2) - float(x1)
    h = float(y2) - float(y1)
    cx = float(x1) + w / 2.0
    cy = float(y1) + h / 2.0
    s = w * h
    r = w / h if h > 1e-9 else 0.0
    return cx, cy, s, r


def _z_to_bbox(cx: float, cy: float, s: float, r: float) -> np.ndarray:
    """(cx, cy, s, r) -> [x1, y1, x2, y2]. Guards against a degenerate/negative
    scale (possible after several unmatched predict() calls) instead of
    letting `sqrt()` raise on negative input."""
    s = max(float(s), 1e-6)
    r = max(float(r), 1e-6)
    w = float(np.sqrt(s * r))
    h = s / w if w > 1e-9 else 0.0
    return np.array(
        [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
        dtype="float64",
    )


class KalmanBoxTracker:
    """SORT state [cx, cy, s, r, vcx, vcy, vs], constant velocity, pure numpy.

    No velocity term for `r` (aspect ratio) — per the original SORT paper's
    formulation, box shape is assumed to change slowly enough that modeling
    its rate of change adds noise rather than signal.

    Covariance/noise tuning (P/Q/R initial values and scaling) mirrors the
    widely-used `filterpy`-based reference SORT implementation: velocities
    start with high uncertainty (unobservable at init), scale/aspect-ratio
    measurement noise is inflated relative to position, and velocity process
    noise is damped so the filter doesn't overreact to a single noisy
    detection.
    """

    def __init__(self, bbox, track_id: int):
        self.track_id = int(track_id)

        cx, cy, s, r = _bbox_to_z(bbox)
        self.x = np.array([cx, cy, s, r, 0.0, 0.0, 0.0], dtype="float64")

        # Constant-velocity state transition: position/scale += velocity.
        self.F = np.eye(7, dtype="float64")
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0

        # We only observe [cx, cy, s, r], never the velocities directly.
        self.H = np.zeros((4, 7), dtype="float64")
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        self.R = np.eye(4, dtype="float64")
        self.R[2:, 2:] *= 10.0  # trust scale/aspect-ratio measurements less

        self.P = np.eye(7, dtype="float64")
        self.P[4:, 4:] *= 1000.0  # velocities: high initial uncertainty
        self.P *= 10.0

        self.Q = np.eye(7, dtype="float64")
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01  # damp velocity process noise

        self.hits = 0
        self.age = 0
        self.time_since_update = 0
        # True once this track has EVER satisfied the reporting gate
        # (hits >= min_hits, or the tracker-startup exception) at least once.
        # Stays True for the track's whole lifetime, even though `hits`
        # itself keeps resetting to 0 on every missed-frame gap -- this is
        # what lets a re-matched "Lost" track report again immediately
        # instead of re-earning min_hits from scratch (real ByteTrack/
        # DeepSORT-family "Lost track re-activation" semantics). Set by
        # `ByteTracker.update()`, not here.
        self.confirmed = False

    def predict(self) -> np.ndarray:
        """Advance the state one frame (no observation yet) and return the
        predicted bbox in xyxy."""
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        if self.time_since_update > 0:
            # Missed at least one update since the last correction -> the
            # consecutive-hits streak breaks.
            self.hits = 0
        self.time_since_update += 1

        return _z_to_bbox(self.x[0], self.x[1], self.x[2], self.x[3])

    def update(self, bbox) -> None:
        """Correct the filter with a new observed bbox."""
        z = np.array(_bbox_to_z(bbox), dtype="float64")

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hits += 1


class ByteTracker:
    """Two-stage (high-confidence / low-confidence) IoU tracker, ByteTrack-style.

    Both stages use `scipy.optimize.linear_sum_assignment` on an IoU cost
    matrix (`cost = 1 - IoU`). Stage 1 matches high-confidence detections
    against every predicted track; stage 2 matches low-confidence detections
    against whatever tracks stage 1 left unmatched, which is the core
    ByteTrack idea — a low-confidence box recovers an occluded track instead
    of being discarded outright (as most trackers do) or spawning a spurious
    new id.
    """

    def __init__(self, high_thresh=0.5, low_thresh=0.1, match_thresh=0.8,
                 max_age=30, min_hits=3):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.min_hits = min_hits

        self.tracks: List[KalmanBoxTracker] = []
        self._next_id = 1
        self.frame_count = 0

    def _associate(self, det_idx, track_idx, dets_xyxy, predicted_bboxes):
        """Hungarian IoU assignment restricted to the given det/track index
        subsets. Returns (matches, unmatched_dets, unmatched_tracks) as lists
        of ORIGINAL indices (into `dets_xyxy`/`self.tracks` respectively)."""
        if not det_idx or not track_idx:
            return [], list(det_idx), list(track_idx)

        cost = np.ones((len(det_idx), len(track_idx)), dtype="float64")
        for di, d in enumerate(det_idx):
            for ti, t in enumerate(track_idx):
                cost[di, ti] = 1.0 - iou(dets_xyxy[d], predicted_bboxes[t])

        row_ind, col_ind = linear_sum_assignment(cost)

        matches = []
        matched_dets = set()
        matched_tracks = set()
        reject_cost = 1.0 - self.match_thresh
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] > reject_cost:
                # IoU < match_thresh -- reject the pair, both sides stay unmatched.
                continue
            d, t = det_idx[r], track_idx[c]
            matches.append((d, t))
            matched_dets.add(d)
            matched_tracks.add(t)

        unmatched_dets = [d for d in det_idx if d not in matched_dets]
        unmatched_tracks = [t for t in track_idx if t not in matched_tracks]
        return matches, unmatched_dets, unmatched_tracks

    def update(self, dets_xyxy: np.ndarray, scores: np.ndarray) -> List[Tuple[int, int]]:
        """Advance every track by one frame and associate this frame's
        detections. Returns `[(det_index, track_id), ...]` for confirmed
        tracks matched THIS frame — unconfirmed tracks and unmatched
        detections are simply absent from the result (never `-1` entries;
        callers distinguish "no id" by absence)."""
        self.frame_count += 1

        dets_xyxy = np.asarray(dets_xyxy, dtype="float64").reshape(-1, 4)
        scores = np.asarray(scores, dtype="float64").reshape(-1)
        n_dets = dets_xyxy.shape[0]

        predicted_bboxes = [trk.predict() for trk in self.tracks]

        high_det_idx = [i for i in range(n_dets) if scores[i] >= self.high_thresh]
        low_det_idx = [
            i for i in range(n_dets)
            if self.low_thresh <= scores[i] < self.high_thresh
        ]
        all_track_idx = list(range(len(self.tracks)))

        # Stage 1: high-confidence detections vs ALL predicted tracks.
        matches_1, unmatched_dets_1, unmatched_tracks_1 = self._associate(
            high_det_idx, all_track_idx, dets_xyxy, predicted_bboxes
        )

        # Stage 2: low-confidence detections vs tracks stage 1 left unmatched.
        matches_2, _unmatched_low_dets, _unmatched_tracks_2 = self._associate(
            low_det_idx, unmatched_tracks_1, dets_xyxy, predicted_bboxes
        )

        det_to_track = {}
        for det_i, trk_i in matches_1 + matches_2:
            det_to_track[det_i] = self.tracks[trk_i]

        for det_i, trk in det_to_track.items():
            trk.update(dets_xyxy[det_i])

        # Unmatched HIGH-confidence detections spawn new tracks. Unmatched
        # low-confidence detections are discarded (ByteTrack semantics) --
        # they never create tracks, only ever recover existing ones.
        for det_i in unmatched_dets_1:
            new_trk = KalmanBoxTracker(dets_xyxy[det_i], self._next_id)
            self._next_id += 1
            new_trk.update(dets_xyxy[det_i])
            self.tracks.append(new_trk)
            det_to_track[det_i] = new_trk

        # Drop stale tracks (no successful update for more than max_age frames).
        self.tracks = [trk for trk in self.tracks if trk.time_since_update <= self.max_age]
        alive_ids = {trk.track_id for trk in self.tracks}

        results = []
        for det_i, trk in det_to_track.items():
            if trk.track_id not in alive_ids:
                continue
            # First-time confirmation gate: hits >= min_hits, or the SORT
            # startup exception (report even with hits < min_hits during the
            # tracker's first min_hits frames, otherwise no track could ever
            # be reported in a video's opening seconds). Once satisfied,
            # latch `trk.confirmed` permanently -- this is the ONLY place a
            # track becomes confirmed.
            if trk.hits >= self.min_hits or self.frame_count <= self.min_hits:
                trk.confirmed = True

            # Real ByteTrack/DeepSORT "Lost track re-activation" semantics: a
            # track that has EVER been confirmed keeps reporting under the
            # same track_id on every successful re-match, with no fresh
            # min_hits delay after an occlusion gap -- only a genuinely NEW
            # (never-yet-confirmed) track goes through the min_hits startup
            # gate. `hits`/`hit_streak` still reset on a miss (unchanged --
            # that bookkeeping is what proves first-time confirmation), but
            # `confirmed` does not, so recovery is reported immediately.
            if trk.confirmed and trk.time_since_update == 0:
                results.append((int(det_i), int(trk.track_id)))

        return sorted(results, key=lambda pair: pair[0])


def assign_track_ids(reid: "YOLOPersonReID", cfg) -> np.ndarray:
    """Sequential pass over `reid.frame_order`; writes `entry["track_id"]`
    onto every person entry in `reid.results_json`; returns an rid-indexed
    label array shaped exactly like `cluster_locally()`'s return value, so
    `enroll_and_assign_global_ids(gallery, labels, ...)` consumes it with no
    changes: `labels[rid] = track_id`. Detections with no `_rid` or
    belonging to an unconfirmed track get `-1`.

    Runs independently of the YOLO inference `batch_size` — `reid.frame_order`
    is the fully-assembled per-frame stream `detect_and_embed()` processed,
    so an inference-batch boundary never truncates a track.
    """
    n_embeddings = len(reid.embeddings)
    labels = np.full(n_embeddings, -1, dtype="int64")

    tracker = ByteTracker(
        high_thresh=cfg["high_thresh"],
        low_thresh=cfg["low_thresh"],
        match_thresh=cfg["match_thresh"],
        max_age=cfg["max_age"],
        min_hits=cfg["min_hits"],
    )

    frame_order = getattr(reid, "frame_order", None)
    if not frame_order:
        frame_order = list(reid.results_json.keys())

    for frame in frame_order:
        entries = reid.results_json.get(frame, [])
        person_entries = [p for p in entries if p.get("kind") == "person"]

        if person_entries:
            dets_xyxy = np.array([p["bbox"] for p in person_entries], dtype="float64")
            scores = np.array(
                [p["confidence"] if p.get("confidence") is not None else 1.0
                 for p in person_entries],
                dtype="float64",
            )
        else:
            dets_xyxy = np.zeros((0, 4), dtype="float64")
            scores = np.zeros((0,), dtype="float64")

        matches = tracker.update(dets_xyxy, scores)
        det_to_track_id = dict(matches)

        for local_i, p in enumerate(person_entries):
            track_id = det_to_track_id.get(local_i, -1)
            p["track_id"] = int(track_id)

            rid = p.get("_rid")
            if rid is not None and track_id != -1:
                rid = int(rid)
                if 0 <= rid < n_embeddings:
                    labels[rid] = int(track_id)

    return labels
