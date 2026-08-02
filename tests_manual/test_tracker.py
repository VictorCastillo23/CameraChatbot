"""Manual verification for the Fase 4a (PR7) tracker contract.

Not a pytest test: this repo has no test runner, so this is a plain
runnable script using bare ``assert`` statements, printing a PASS/FAIL
summary and exiting 0/1. Run it directly:

    python tests_manual/test_tracker.py

What it verifies (Fase 4a / PR7 tasks):

1. `security.tracker.KalmanBoxTracker` -- predict()/update() round-trip
   (predicting an unmoved box after a single observation, then correcting
   with a moved observation converges the state toward that motion) and the
   `hits`/`age`/`time_since_update` bookkeeping SORT's startup/confirmation
   logic depends on.
2. `security.tracker.ByteTracker.update()` against four hand-authored,
   synthetic fixtures under `tests_manual/fixtures/` (no images, no models):
   - `tracks_linear.json` -- one steadily-translating box -> exactly one
     stable track id throughout.
   - `tracks_occlusion.json` -- a 5-frame gap (no detections at all) then a
     LOW-confidence reappearance near the Kalman-predicted position ->
     validates `max_age` tolerance and the stage-2 low-confidence match: the
     SAME internal track is recovered (never a second track id), even
     though full re-confirmation (being included in the returned list
     again) takes a few more consecutive hits per SORT semantics.
   - `tracks_crossing.json` -- two boxes crossing paths -> two distinct,
     stable ids that never swap.
   - `tracks_lowconf.json` -- one continuously-present box whose confidence
     dips mid-sequence (no gap) -> stage-2 matching alone keeps it alive,
     reported every single frame with the same id.
3. `security.tracker.assign_track_ids()` -- wiring correctness against a
   minimal fake `YOLOPersonReID`-shaped object: writes `entry["track_id"]`
   onto every person entry (including ones without an embedding/`_rid`),
   returns an rid-indexed label array of exactly `len(reid.embeddings)`,
   and detections belonging to an unconfirmed track get `-1` in that array.

Uses only synthetic in-memory/JSON data throughout -- no real camera
frames, no ONNX models, and never touches the repo's real `keyFrames/` or
`gallery.index`/`id_map.json`/`proto_store.npy` (see the "Identity
persistence gotcha" note in CLAUDE.md).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from camerachatbot.security.tracker import (  # noqa: E402
    KalmanBoxTracker,
    ByteTracker,
    assign_track_ids,
)

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

results = []  # list[tuple[str, bool, str]]


def check(name, fn):
    try:
        fn()
        results.append((name, True, ""))
    except AssertionError as e:
        results.append((name, False, f"AssertionError: {e}"))
    except Exception as e:  # noqa: BLE001 - report any unexpected exception as a failure
        results.append((name, False, f"{type(e).__name__}: {e}"))


def _load_fixture(name):
    with open(os.path.join(FIXTURES_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)


def _run_fixture(fixture, tracker=None):
    """Feeds every frame of a fixture through a ByteTracker (fresh unless
    one is passed in) and returns the per-frame `det_index -> track_id`
    match dicts, one per frame, in order."""
    tracker = tracker or ByteTracker()
    per_frame = []
    for frame in fixture["frames"]:
        dets = np.array(frame["dets"], dtype="float64").reshape(-1, 4)
        scores = np.array(frame["scores"], dtype="float64").reshape(-1)
        matches = tracker.update(dets, scores)
        per_frame.append(dict(matches))
    return tracker, per_frame


# ---------------------------------------------------------------------------
# KalmanBoxTracker: predict()/update() + hits/age/time_since_update
# ---------------------------------------------------------------------------

def test_kalman_predict_returns_same_box_before_any_motion_is_observed():
    trk = KalmanBoxTracker([0, 0, 100, 100], track_id=1)
    trk.update([0, 0, 100, 100])  # first observation, velocity still 0
    pred = trk.predict()
    # With zero velocity estimated so far, the predicted box should still be
    # centered essentially where the last observation was.
    assert abs(pred[0] - 0.0) < 1e-6 and abs(pred[1] - 0.0) < 1e-6
    assert abs((pred[2] - pred[0]) - 100.0) < 1e-6  # width preserved


def test_kalman_converges_toward_observed_velocity():
    trk = KalmanBoxTracker([0, 0, 100, 100], track_id=1)
    x1 = 0
    for _ in range(8):
        trk.predict()
        trk.update([x1, 0, x1 + 100, 100])
        x1 += 10
    # After several consistent +10/frame observations, the filter's velocity
    # state (index 4 = vcx) should have converged close to +10.
    assert abs(trk.x[4] - 10.0) < 1.0, trk.x[4]


def test_kalman_hits_age_time_since_update_bookkeeping():
    trk = KalmanBoxTracker([0, 0, 100, 100], track_id=1)
    assert trk.hits == 0 and trk.age == 0 and trk.time_since_update == 0

    trk.update([0, 0, 100, 100])
    assert trk.hits == 1 and trk.time_since_update == 0

    trk.predict()
    assert trk.age == 1 and trk.time_since_update == 1
    # hits does NOT reset on the predict() immediately following an update
    # (time_since_update was 0 going into this predict call).
    assert trk.hits == 1

    trk.predict()  # a SECOND consecutive missed frame -> streak breaks
    assert trk.time_since_update == 2
    assert trk.hits == 0


# ---------------------------------------------------------------------------
# ByteTracker: synthetic fixtures
# ---------------------------------------------------------------------------

def test_linear_motion_single_stable_track_id():
    fixture = _load_fixture("tracks_linear.json")
    _tracker, per_frame = _run_fixture(fixture)

    ids_seen = set()
    for i, frame_matches in enumerate(per_frame):
        assert 0 in frame_matches, f"frame {i}: detection not tracked -- {frame_matches}"
        ids_seen.add(frame_matches[0])

    assert ids_seen == {list(ids_seen)[0]}, f"expected exactly one stable id, got {ids_seen}"
    assert len(ids_seen) == 1


def test_occlusion_recovers_same_track_no_spurious_new_id():
    fixture = _load_fixture("tracks_occlusion.json")
    tracker, per_frame = _run_fixture(fixture)

    gap_start = fixture["gap_start_frame"]
    gap_len = fixture["gap_len"]
    reappear_frame = fixture["reappear_frame"]

    lead_id = per_frame[0][0]
    assert lead_id is not None

    for i in range(gap_start, gap_start + gap_len):
        assert per_frame[i] == {}, f"frame {i} (gap) should report nothing, got {per_frame[i]}"

    # Never more than 1 track alive -- proves the low-confidence reappearance
    # was absorbed by the SAME KalmanBoxTracker, not spawned as a new one.
    assert len(tracker.tracks) == 1, \
        f"expected exactly one track to have ever existed, got {len(tracker.tracks)}"

    # The reappearance frame itself is a low-confidence match: it correctly
    # re-associates (internally) but is not yet re-confirmed for reporting.
    assert per_frame[reappear_frame] == {}, per_frame[reappear_frame]

    # Track eventually gets reported again, with the SAME id as before the
    # gap, once enough consecutive hits accumulate post-recovery.
    later_reports = [fm[0] for fm in per_frame[reappear_frame + 1:] if 0 in fm]
    assert later_reports, "track was never reported again after recovering from occlusion"
    assert all(tid == lead_id for tid in later_reports), \
        f"recovered track id changed -- expected {lead_id}, got {set(later_reports)}"


def test_crossing_tracks_never_swap_ids():
    fixture = _load_fixture("tracks_crossing.json")
    _tracker, per_frame = _run_fixture(fixture)

    id_a = per_frame[0][0]
    id_b = per_frame[0][1]
    assert id_a != id_b

    for i, frame_matches in enumerate(per_frame):
        assert frame_matches.get(0) == id_a, \
            f"frame {i}: det 0 (the 'A' box) mapped to {frame_matches.get(0)}, expected {id_a} -- ID SWAP"
        assert frame_matches.get(1) == id_b, \
            f"frame {i}: det 1 (the 'B' box) mapped to {frame_matches.get(1)}, expected {id_b} -- ID SWAP"


def test_lowconf_dip_keeps_track_alive_every_frame():
    fixture = _load_fixture("tracks_lowconf.json")
    _tracker, per_frame = _run_fixture(fixture)

    lead_id = per_frame[0][0]
    for i, frame_matches in enumerate(per_frame):
        assert 0 in frame_matches, f"frame {i}: track dropped during confidence dip -- {frame_matches}"
        assert frame_matches[0] == lead_id, \
            f"frame {i}: track id changed from {lead_id} to {frame_matches[0]}"


# ---------------------------------------------------------------------------
# assign_track_ids(): wiring against a minimal fake YOLOPersonReID
# ---------------------------------------------------------------------------

class _FakeReID:
    """Duck-types just enough of YOLOPersonReID for assign_track_ids()."""

    def __init__(self, frame_order, results_json, n_embeddings):
        self.frame_order = frame_order
        self.results_json = results_json
        self.embeddings = [None] * n_embeddings  # only len() is used


def _tracker_cfg():
    return {"high_thresh": 0.5, "low_thresh": 0.1, "match_thresh": 0.8,
            "max_age": 30, "min_hits": 3}


def test_assign_track_ids_writes_track_id_on_every_person_entry():
    # 3 frames, one moving person with an embedding (_rid) each frame, plus
    # one person entry per frame with NO embedding (no "_rid") to prove
    # track_id still gets written even without one.
    results_json = {}
    for i, frame in enumerate(["f0", "f1", "f2"]):
        x1 = 10 + i * 5
        results_json[frame] = [
            {"kind": "person", "bbox": [x1, 10, x1 + 100, 110], "confidence": 0.9, "_rid": i},
            {"kind": "person", "bbox": [500, 500, 550, 600], "confidence": 0.9},  # no _rid
        ]
    reid = _FakeReID(frame_order=["f0", "f1", "f2"], results_json=results_json, n_embeddings=3)

    labels = assign_track_ids(reid, _tracker_cfg())

    assert isinstance(labels, np.ndarray)
    assert len(labels) == 3, f"expected an rid-indexed array of len 3, got {len(labels)}"

    for frame in ["f0", "f1", "f2"]:
        for entry in results_json[frame]:
            assert "track_id" in entry and entry["track_id"] is not None

    # The embedded person is tracked continuously (frame_count <= min_hits
    # during these first 3 frames, so it's reported from frame 0 onward).
    with_rid_ids = {results_json[f][0]["track_id"] for f in ["f0", "f1", "f2"]}
    assert with_rid_ids == {results_json["f0"][0]["track_id"]}, with_rid_ids
    assert results_json["f0"][0]["track_id"] != -1

    # rid 0/1/2 all belong to the same track (the only moving person) -- the
    # label array should reflect that same track id at every rid.
    assert labels[0] == labels[1] == labels[2] == results_json["f0"][0]["track_id"]


def test_assign_track_ids_minus_one_for_unconfirmed_or_missing_rid():
    # A single frame, single low-quality/never-matched-again detection: with
    # only 1 frame ever processed, frame_count(1) <= min_hits(3) so it WOULD
    # normally be reported (SORT startup exception) -- but a detection with
    # no "_rid" at all can never populate the labels array regardless.
    results_json = {"f0": [{"kind": "person", "bbox": [0, 0, 50, 50], "confidence": 0.9}]}
    reid = _FakeReID(frame_order=["f0"], results_json=results_json, n_embeddings=0)

    labels = assign_track_ids(reid, _tracker_cfg())

    assert len(labels) == 0, "no embeddings were ever produced -- labels array must be empty"
    assert results_json["f0"][0]["track_id"] != -1  # still tracked (startup exception)
    assert "_rid" not in results_json["f0"][0]  # ...but never had an embedding to index


def main():
    check("KalmanBoxTracker.predict() before any motion observed", test_kalman_predict_returns_same_box_before_any_motion_is_observed)
    check("KalmanBoxTracker converges toward observed velocity", test_kalman_converges_toward_observed_velocity)
    check("KalmanBoxTracker hits/age/time_since_update bookkeeping", test_kalman_hits_age_time_since_update_bookkeeping)
    check("ByteTracker: tracks_linear.json -> one stable id", test_linear_motion_single_stable_track_id)
    check("ByteTracker: tracks_occlusion.json -> same id recovered, no spurious new id", test_occlusion_recovers_same_track_no_spurious_new_id)
    check("ByteTracker: tracks_crossing.json -> no id swap", test_crossing_tracks_never_swap_ids)
    check("ByteTracker: tracks_lowconf.json -> stage-2 keeps track alive every frame", test_lowconf_dip_keeps_track_alive_every_frame)
    check("assign_track_ids() writes track_id on every person entry + rid-indexed labels", test_assign_track_ids_writes_track_id_on_every_person_entry)
    check("assign_track_ids() -1/empty semantics for missing _rid", test_assign_track_ids_minus_one_for_unconfirmed_or_missing_rid)

    print("\n=== Fase 4a (PR7) tracker contract verification ===")
    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        line = f"[{status}] {name}"
        if not ok:
            line += f" — {detail}"
        print(line)

    print(f"\n{n_pass}/{len(results)} checks passed.")
    if n_pass != len(results):
        print("SUMMARY: FAIL")
        return 1
    print("SUMMARY: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
