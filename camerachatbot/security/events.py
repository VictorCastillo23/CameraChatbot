"""Fase 4b (PR8a/PR8b) + Fase 5 (PR9) — tracks-timeline construction and
intrusion/loitering/unenrolled-person rules.

`build_tracks_timeline()` is the single most important testability decision
in this phase: every event-generating rule is a PURE function over one
intermediate structure (`TracksTimeline`), built once. That lets every rule
be tested with hand-built dicts and zero image I/O (see `tests_manual/`).

This module is pure logic only — no database writes, no orchestrator
wiring (that lands in `pipeline.orchestrator._run_zones_events_stage()`).
`evaluate_unenrolled` (Fase 5, PR9) is implemented here now; `evaluate_
weapons` (Fase 6) is still deliberately NOT implemented — it will be added
to this same file by a later PR.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, TypedDict

from camerachatbot.geometry.homography import CameraCalibration
from camerachatbot.identity.authorization import AuthorizationRegistry
from camerachatbot.security.zones import Zone, bbox_world_and_zone, zone_is_armed
from camerachatbot.video_schema.timing import frame_timestamp


class TrackObservation(TypedDict):
    track_id: int
    frame: str
    frame_idx: int
    ts: datetime
    bbox: List[int]
    world_xy: Optional[Tuple[float, float]]
    zone_id: Optional[int]
    person_global_id: Optional[int]
    confidence: float


TracksTimeline = Dict[int, List[TrackObservation]]  # per track, frame-ordered


@dataclass
class SecurityEvent:
    event_type: str  # intrusion | loitering | unenrolled_person | weapons_detected
    zone_id: Optional[int]
    person_global_id: Optional[int]
    track_id: Optional[int]
    started_at: datetime
    ended_at: Optional[datetime]
    first_frame_idx: int
    confidence: Optional[float]
    details: dict = field(default_factory=dict)


def event_to_dict(event: SecurityEvent) -> dict:
    """JSON/Postgres-safe serialization of a `SecurityEvent` (PR8b).

    Shared by `orchestrator.multi_models()`'s `security_events.json`
    checkpoint dump and `video_schema.formatter.reformat_to_video_schema_
    uniform()`'s `video.events` node -- one implementation instead of two
    that could drift on timestamp formatting (same anti-drift principle as
    `video_schema.timing.frame_timestamp`/`parse_start_at`).

    `started_at`/`ended_at` are ISO-8601 strings, matching the convention
    `key_frame.timestamp` already uses for a Postgres TIMESTAMP column (see
    `db.postgres_writer.prepare_keyframe_rows`, which inserts `kf["timestamp"]`
    -- also an ISO string -- directly).
    """
    return {
        "event_type": event.event_type,
        "zone_id": event.zone_id,
        "person_global_id": event.person_global_id,
        "track_id": event.track_id,
        "started_at": event.started_at.isoformat() if event.started_at else None,
        "ended_at": event.ended_at.isoformat() if event.ended_at else None,
        "first_frame_idx": event.first_frame_idx,
        "confidence": event.confidence,
        "details": event.details,
    }


def _frame_idx_of(frame: str, fallback: int) -> int:
    """`int(frame)` when the frame name is numeric (the normal case — frame
    names are the numeric filename stem), else `fallback` (its position in
    `frame_order`). Falling back rather than raising keeps
    `build_tracks_timeline()` usable even with non-numeric frame names,
    though `frame_timestamp()`'s alignment with `key_frame.timestamp` rows
    then only holds when `formatter.py` degrades the same way."""
    try:
        return int(frame)
    except (TypeError, ValueError):
        return fallback


def build_tracks_timeline(
    reid,
    calib: Optional[CameraCalibration],
    zones: List[Zone],
    t0: datetime,
    fps: float,
) -> TracksTimeline:
    """Reads `reid.results_json` (bboxes, `track_id`, `person_global_id`)
    and `reid.frame_order`, builds one `TrackObservation` per tracked person
    detection, grouped by `track_id`, frame-ordered.

    Degrades gracefully when `calib is None` (no camera calibration yet):
    `world_xy`/`zone_id` are `None` for every observation instead of
    raising, so tracking and `person_global_id` stay usable — and Fase 5's
    `unenrolled` events usable — before any camera is calibrated.

    Detections with no track (`track_id` missing, `None`, or `-1`, per
    `security.tracker.assign_track_ids()`'s "no id" convention) are skipped.
    """
    timeline: TracksTimeline = {}

    frame_order = getattr(reid, "frame_order", None) or list(reid.results_json.keys())

    for pos, frame in enumerate(frame_order):
        entries = reid.results_json.get(frame, [])
        frame_idx = _frame_idx_of(frame, pos)
        ts = frame_timestamp(t0, frame_idx, fps)

        for p in entries:
            if p.get("kind") != "person":
                continue
            track_id = p.get("track_id")
            if track_id is None or track_id == -1:
                continue

            bbox = p.get("bbox")
            world_xy = None
            zone_id = None
            if calib is not None and bbox is not None:
                # PR8b: `bbox_world_and_zone()` is the single shared
                # bbox->world->zone computation -- `orchestrator.
                # _write_zone_and_world_xy()` uses the exact same helper for
                # the entries themselves, so a TrackObservation's world_xy/
                # zone_id can never drift out of sync with what's written
                # onto `p["world_xy"]`/`p["zone_id"]`.
                world_xy, zone = bbox_world_and_zone(bbox, calib, zones)
                zone_id = zone.id if zone is not None else None

            confidence = p.get("confidence")
            obs: TrackObservation = {
                "track_id": int(track_id),
                "frame": frame,
                "frame_idx": frame_idx,
                "ts": ts,
                "bbox": list(bbox) if bbox is not None else [],
                "world_xy": world_xy,
                "zone_id": zone_id,
                "person_global_id": p.get("person_global_id"),
                "confidence": float(confidence) if confidence is not None else 0.0,
            }
            timeline.setdefault(int(track_id), []).append(obs)

    return timeline


def _runs(
    observations: List[TrackObservation],
    predicate: Callable[[TrackObservation], bool],
    gap_tolerance: int,
) -> List[List[TrackObservation]]:
    """Split `observations` (already frame-ordered) into maximal runs of
    consecutive observations satisfying `predicate`, tolerating gaps of up
    to `gap_tolerance` frames (by `frame_idx`) between two qualifying
    observations — so a single dropped detection doesn't split one
    continuous event into several rows.

    Shared by `evaluate_intrusion` and `evaluate_loitering` so both rules
    use the exact same "continuous run" semantics.
    """
    runs: List[List[TrackObservation]] = []
    current: List[TrackObservation] = []

    for obs in observations:
        if not predicate(obs):
            continue
        if current and (obs["frame_idx"] - current[-1]["frame_idx"]) > gap_tolerance + 1:
            runs.append(current)
            current = []
        current.append(obs)

    if current:
        runs.append(current)

    return runs


def evaluate_intrusion(
    timeline: TracksTimeline,
    zones: List[Zone],
    cfg: dict,
) -> List[SecurityEvent]:
    """One `SecurityEvent(event_type="intrusion")` per maximal run where the
    observation's zone is `restricted` AND currently armed
    (`zone_is_armed(zone, obs.ts)`), tolerating `cfg["intrusion_gap_frames"]`
    frame gaps within one run."""
    zones_by_id = {z.id: z for z in zones}
    gap = cfg["intrusion_gap_frames"]
    events: List[SecurityEvent] = []

    def _is_intrusion(obs: TrackObservation) -> bool:
        zone = zones_by_id.get(obs["zone_id"]) if obs["zone_id"] is not None else None
        if zone is None or zone.zone_type != "restricted":
            return False
        return zone_is_armed(zone, obs["ts"])

    for track_id, observations in timeline.items():
        for run in _runs(observations, _is_intrusion, gap):
            first, last = run[0], run[-1]
            zone = zones_by_id.get(first["zone_id"])
            events.append(SecurityEvent(
                event_type="intrusion",
                zone_id=first["zone_id"],
                person_global_id=first.get("person_global_id"),
                track_id=track_id,
                started_at=first["ts"],
                ended_at=last["ts"],
                first_frame_idx=first["frame_idx"],
                confidence=None,
                details={
                    "zone_name": zone.name if zone is not None else None,
                    "observation_count": len(run),
                },
            ))

    return events


def evaluate_loitering(
    timeline: TracksTimeline,
    zones: List[Zone],
    cfg: dict,
) -> List[SecurityEvent]:
    """One `SecurityEvent(event_type="loitering")` per zone whose effective
    `loiter_seconds` is set, per maximal run (same `_runs()` gap tolerance
    as `evaluate_intrusion`) spent inside that zone, emitted only when the
    run's dwell time (`ended_at - started_at`) is `>= loiter_seconds`.

    Effective `loiter_seconds` is `zone.schedule.get("loiter_seconds")`,
    falling back to `cfg["default_loiter_seconds"]` when the zone has no
    schedule or the schedule doesn't set it.
    """
    gap = cfg["intrusion_gap_frames"]
    default_loiter = cfg["default_loiter_seconds"]
    events: List[SecurityEvent] = []

    for zone in zones:
        loiter_seconds = (zone.schedule or {}).get("loiter_seconds", default_loiter)
        if loiter_seconds is None:
            continue

        def _in_zone(obs: TrackObservation, _zone_id: int = zone.id) -> bool:
            return obs["zone_id"] == _zone_id

        for track_id, observations in timeline.items():
            for run in _runs(observations, _in_zone, gap):
                first, last = run[0], run[-1]
                dwell_seconds = (last["ts"] - first["ts"]).total_seconds()
                if dwell_seconds >= loiter_seconds:
                    events.append(SecurityEvent(
                        event_type="loitering",
                        zone_id=zone.id,
                        person_global_id=first.get("person_global_id"),
                        track_id=track_id,
                        started_at=first["ts"],
                        ended_at=last["ts"],
                        first_frame_idx=first["frame_idx"],
                        confidence=None,
                        details={
                            "zone_name": zone.name,
                            "dwell_seconds": dwell_seconds,
                            "observation_count": len(run),
                        },
                    ))

    return events


def evaluate_unenrolled(
    timeline: TracksTimeline,
    registry: AuthorizationRegistry,
    cfg: dict,
) -> List[SecurityEvent]:
    """One `SecurityEvent(event_type="unenrolled_person")` per maximal run
    (Fase 5) where the observation's `person_global_id` is either `None` or
    not authorized per `registry.is_authorized(person_global_id)`.

    A run is only reported once it reaches `>= cfg["unenrolled_debounce_
    frames"]` OBSERVATIONS. This is deliberately a count of observations,
    NOT wall-clock time -- a low-fps run must not arm faster than a
    high-fps one just because more real time elapses between frames.

    Uses the same `_runs()` gap-tolerance primitive
    (`cfg["intrusion_gap_frames"]`) as `evaluate_intrusion`/`evaluate_
    loitering`, so a single dropped detection does not fragment one
    continuous unenrolled presence into multiple under-threshold runs.

    `registry` is camera-independent (see `identity.authorization.
    AuthorizationRegistry`) and fail-closed: `is_authorized(None)` and any
    unknown/deauthorized `person_global_id` both return `False`, which is
    exactly what makes an unenrolled/deauthorized person's presence
    detectable here.
    """
    gap = cfg["intrusion_gap_frames"]
    debounce = cfg["unenrolled_debounce_frames"]
    events: List[SecurityEvent] = []

    def _is_unenrolled(obs: TrackObservation) -> bool:
        pid = obs.get("person_global_id")
        return pid is None or not registry.is_authorized(pid)

    for track_id, observations in timeline.items():
        for run in _runs(observations, _is_unenrolled, gap):
            if len(run) < debounce:
                continue
            first, last = run[0], run[-1]
            events.append(SecurityEvent(
                event_type="unenrolled_person",
                zone_id=first["zone_id"],
                person_global_id=first.get("person_global_id"),
                track_id=track_id,
                started_at=first["ts"],
                ended_at=last["ts"],
                first_frame_idx=first["frame_idx"],
                confidence=None,
                details={
                    "observation_count": len(run),
                },
            ))

    return events
