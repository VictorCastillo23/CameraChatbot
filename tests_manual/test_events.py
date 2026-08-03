"""Manual verification for the Fase 4b (PR8a) zones+events contract.

Not a pytest test: this repo has no test runner, so this is a plain
runnable script using bare ``assert`` statements, printing a PASS/FAIL
summary and exiting 0/1. Run it directly:

    python tests_manual/test_events.py

Split from `test_zones.py` because `zones.py` and `events.py` are two
substantial, independently-testable modules (zones.py: geometry + schedule
+ DB loading; events.py: timeline construction + rule evaluation) -- same
one-test-file-per-module precedent as `test_tracker.py`. `test_events.py`
still imports `security.zones.Zone` directly since events.py's evaluators
take `list[Zone]`, but does not re-test zones.py's own logic.

What it verifies (Fase 4b / PR8a task 4b.1 + 4b.3):

1. `video_schema.timing.frame_timestamp()` -- the extracted shared
   arithmetic (`t0 + idx/fps`), including the exact SAME formula
   `formatter.py` now delegates to (task 4b.1's whole point: one
   implementation, not two that could drift).
2. `security.events.build_tracks_timeline()` against a fake
   `YOLOPersonReID`-shaped object: groups by `track_id`, frame-ordered;
   skips non-person entries and untracked detections (`track_id` missing/
   `None`/`-1`); computes `ts` via `frame_timestamp()` using `int(frame)`
   as `idx` (the SAME convention `formatter.py` uses for `key_frame.
   timestamp`, so events line up with keyframe rows); degrades to
   `world_xy=None`/`zone_id=None` with `calib=None` instead of crashing;
   classifies `zone_id` via `classify_bbox_zone()` when a calibration IS
   available.
3. `security.events.evaluate_intrusion()` -- `SECURITY_RULES` actually has
   an `intrusion_gap_frames` key (confirmed via direct import, not assumed)
   and the shared `_runs()` gap-tolerance primitive works via the public
   evaluator: a one-frame gap inside `intrusion_gap_frames` tolerance stays
   ONE event; a gap exceeding it splits into TWO. Also: a `monitored` zone
   and an unarmed `restricted` zone never produce an intrusion event.
4. `security.events.evaluate_loitering()` -- `SECURITY_RULES` actually has
   a `default_loiter_seconds` key; dwell time just below the zone's
   effective `loiter_seconds` produces zero events, just at/above produces
   one.

Uses only synthetic in-memory/JSON data throughout -- no real camera
frames, no ONNX models, no live Postgres connection, and never touches the
repo's real `keyFrames/`, `res/`, or `gallery.index`/`id_map.json`/
`proto_store.npy`.
"""

import json
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camerachatbot.video_schema.timing import frame_timestamp  # noqa: E402
from camerachatbot.security.zones import Zone  # noqa: E402
from camerachatbot.security.events import (  # noqa: E402
    build_tracks_timeline,
    evaluate_intrusion,
    evaluate_loitering,
)
from camerachatbot.security_config import SECURITY_RULES  # noqa: E402

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


def _zone_from_dict(d):
    return Zone(
        id=d["id"], camera_id=d["camera_id"], name=d["name"], zone_type=d["zone_type"],
        polygon=[tuple(p) for p in d["polygon"]], schedule=d["schedule"], is_active=d["is_active"],
    )


# ---------------------------------------------------------------------------
# video_schema.timing.frame_timestamp(): the shared arithmetic
# ---------------------------------------------------------------------------

def test_frame_timestamp_basic_arithmetic():
    t0 = datetime(2026, 1, 1, 0, 0, 0)
    assert frame_timestamp(t0, 0, fps=1.0) == t0
    assert frame_timestamp(t0, 10, fps=1.0) == t0 + timedelta(seconds=10)
    assert frame_timestamp(t0, 30, fps=30.0) == t0 + timedelta(seconds=1)


def test_frame_timestamp_matches_formatter_style_derivation():
    # Same formula formatter.py used to compute inline before task 4b.1
    # extracted it -- proves this IS that exact arithmetic, not a
    # similar-but-different reimplementation.
    t0 = datetime(2026, 3, 15, 8, 30, 0)
    fps = 25.0
    for idx in (0, 1, 7, 24, 100):
        expected = t0 + timedelta(seconds=(idx / fps))
        assert frame_timestamp(t0, idx, fps) == expected


# ---------------------------------------------------------------------------
# build_tracks_timeline(): fake YOLOPersonReID-shaped object
# ---------------------------------------------------------------------------

class _FakeReID:
    def __init__(self, frame_order, results_json):
        self.frame_order = frame_order
        self.results_json = results_json


class _FakeCalib:
    """Duck-types just enough of CameraCalibration: a trivial 0.01
    pixel->metre scale, bottom-center ground point (same convention as the
    real class)."""

    def bbox_to_world(self, bbox):
        x1, y1, x2, y2 = bbox
        return (0.5 * (x1 + x2) * 0.01, float(y2) * 0.01)


def _covering_zone():
    # World (0,0)-(100,100), covers any ground point from small pixel bboxes
    # scaled by 0.01.
    return Zone(id=5, camera_id=1, name="Covers-All", zone_type="restricted",
                polygon=[(0, 0), (100, 0), (100, 100), (0, 100)], schedule=None, is_active=True)


def test_build_tracks_timeline_skips_non_person_and_untracked_entries():
    results_json = {
        "0": [
            {"kind": "object", "class_name": "knife", "bbox": [0, 0, 10, 10], "confidence": 0.9, "track_id": 1},
            {"kind": "person", "bbox": [0, 0, 10, 10], "confidence": 0.9, "track_id": None, "person_global_id": None},
            {"kind": "person", "bbox": [0, 0, 10, 10], "confidence": 0.9, "track_id": -1, "person_global_id": None},
            {"kind": "person", "bbox": [0, 0, 10, 10], "confidence": 0.9, "track_id": 7, "person_global_id": 42},
        ],
    }
    reid = _FakeReID(frame_order=["0"], results_json=results_json)
    t0 = datetime(2026, 1, 1, 0, 0, 0)

    timeline = build_tracks_timeline(reid, calib=None, zones=[], t0=t0, fps=1.0)

    assert list(timeline.keys()) == [7], f"expected only track_id 7 to survive, got {list(timeline.keys())}"
    assert len(timeline[7]) == 1
    assert timeline[7][0]["person_global_id"] == 42


def test_build_tracks_timeline_frame_idx_and_ts_use_int_frame():
    results_json = {
        "12": [{"kind": "person", "bbox": [0, 0, 10, 10], "confidence": 0.8, "track_id": 1, "person_global_id": None}],
    }
    reid = _FakeReID(frame_order=["12"], results_json=results_json)
    t0 = datetime(2026, 1, 1, 0, 0, 0)
    fps = 5.0

    timeline = build_tracks_timeline(reid, calib=None, zones=[], t0=t0, fps=fps)

    obs = timeline[1][0]
    assert obs["frame_idx"] == 12
    assert obs["ts"] == frame_timestamp(t0, 12, fps), \
        "ts must be derived via the SAME frame_timestamp() formatter.py uses"


def test_build_tracks_timeline_degrades_gracefully_without_calibration():
    results_json = {
        "0": [{"kind": "person", "bbox": [0, 0, 10, 10], "confidence": 0.9, "track_id": 3, "person_global_id": None}],
    }
    reid = _FakeReID(frame_order=["0"], results_json=results_json)
    t0 = datetime(2026, 1, 1, 0, 0, 0)

    timeline = build_tracks_timeline(reid, calib=None, zones=[_covering_zone()], t0=t0, fps=1.0)

    obs = timeline[3][0]
    assert obs["world_xy"] is None
    assert obs["zone_id"] is None


def test_build_tracks_timeline_groups_by_track_id_frame_ordered():
    results_json = {
        "0": [{"kind": "person", "bbox": [0, 0, 10, 10], "confidence": 0.9, "track_id": 1, "person_global_id": None}],
        "1": [{"kind": "person", "bbox": [1, 0, 11, 10], "confidence": 0.9, "track_id": 1, "person_global_id": None}],
        "2": [{"kind": "person", "bbox": [2, 0, 12, 10], "confidence": 0.9, "track_id": 1, "person_global_id": None}],
    }
    reid = _FakeReID(frame_order=["0", "1", "2"], results_json=results_json)
    t0 = datetime(2026, 1, 1, 0, 0, 0)

    timeline = build_tracks_timeline(reid, calib=None, zones=[], t0=t0, fps=1.0)

    assert list(timeline.keys()) == [1]
    frames_seen = [obs["frame"] for obs in timeline[1]]
    assert frames_seen == ["0", "1", "2"], f"expected frame-ordered observations, got {frames_seen}"


def test_build_tracks_timeline_classifies_zone_with_calibration():
    zone = _covering_zone()
    results_json = {
        "0": [{"kind": "person", "bbox": [0, 0, 10, 10], "confidence": 0.9, "track_id": 9, "person_global_id": None}],
    }
    reid = _FakeReID(frame_order=["0"], results_json=results_json)
    t0 = datetime(2026, 1, 1, 0, 0, 0)

    timeline = build_tracks_timeline(reid, calib=_FakeCalib(), zones=[zone], t0=t0, fps=1.0)

    obs = timeline[9][0]
    assert obs["zone_id"] == zone.id, f"expected zone_id={zone.id}, got {obs['zone_id']}"
    assert obs["world_xy"] == (0.05, 0.10), obs["world_xy"]  # bbox [0,0,10,10] -> ground (5,10) * 0.01


# ---------------------------------------------------------------------------
# evaluate_intrusion(): fixture-driven gap tolerance + predicate correctness
# ---------------------------------------------------------------------------

def _observations_from_frame_indices(track_id, zone_id, frame_indices, fps=1.0, t0=None):
    t0 = t0 or datetime(2026, 1, 1, 0, 0, 0)
    obs_list = []
    for idx in frame_indices:
        obs_list.append({
            "track_id": track_id,
            "frame": str(idx),
            "frame_idx": idx,
            "ts": frame_timestamp(t0, idx, fps),
            "bbox": [0, 0, 10, 10],
            "world_xy": None,
            "zone_id": zone_id,
            "person_global_id": None,
            "confidence": 0.9,
        })
    return obs_list


def test_security_rules_has_intrusion_and_loitering_config_fields():
    assert "intrusion_gap_frames" in SECURITY_RULES, "SECURITY_RULES is missing intrusion_gap_frames"
    assert "default_loiter_seconds" in SECURITY_RULES, "SECURITY_RULES is missing default_loiter_seconds"
    assert isinstance(SECURITY_RULES["intrusion_gap_frames"], int)
    assert isinstance(SECURITY_RULES["default_loiter_seconds"], (int, float))


def test_evaluate_intrusion_one_frame_gap_is_one_event():
    fixture = _load_fixture("timeline_intrusion.json")
    zones = [_zone_from_dict(z) for z in fixture["zones"]]
    case = fixture["one_frame_gap"]
    cfg = {"intrusion_gap_frames": fixture["gap_tolerance"], "default_loiter_seconds": 30}

    observations = _observations_from_frame_indices(case["track_id"], zones[0].id, case["frame_indices"])
    timeline = {case["track_id"]: observations}

    events = evaluate_intrusion(timeline, zones, cfg)

    assert len(events) == case["expected_event_count"], \
        f"expected {case['expected_event_count']} event(s), got {len(events)}: {events}"
    ev = events[0]
    assert ev.event_type == "intrusion"
    assert ev.track_id == case["track_id"]
    assert ev.zone_id == zones[0].id
    assert ev.first_frame_idx == case["frame_indices"][0]
    assert ev.started_at == frame_timestamp(datetime(2026, 1, 1, 0, 0, 0), case["frame_indices"][0], 1.0)
    assert ev.ended_at == frame_timestamp(datetime(2026, 1, 1, 0, 0, 0), case["frame_indices"][-1], 1.0)


def test_evaluate_intrusion_exceeding_gap_is_two_events():
    fixture = _load_fixture("timeline_intrusion.json")
    zones = [_zone_from_dict(z) for z in fixture["zones"]]
    case = fixture["exceeding_gap"]
    cfg = {"intrusion_gap_frames": fixture["gap_tolerance"], "default_loiter_seconds": 30}

    observations = _observations_from_frame_indices(case["track_id"], zones[0].id, case["frame_indices"])
    timeline = {case["track_id"]: observations}

    events = evaluate_intrusion(timeline, zones, cfg)

    assert len(events) == case["expected_event_count"], \
        f"expected {case['expected_event_count']} event(s), got {len(events)}: {events}"
    # The two runs must NOT overlap in frame span -- first run ends at frame
    # 14, second run starts at frame 18 (real split, not a duplicate count).
    events_sorted = sorted(events, key=lambda e: e.first_frame_idx)
    assert events_sorted[0].first_frame_idx == 10
    assert events_sorted[1].first_frame_idx == 18


def test_evaluate_intrusion_ignores_non_restricted_zone():
    fixture = _load_fixture("timeline_intrusion.json")
    cfg = {"intrusion_gap_frames": 2, "default_loiter_seconds": 30}
    monitored_zone = Zone(id=1, camera_id=1, name="Lobby", zone_type="monitored",
                           polygon=[], schedule=None, is_active=True)

    observations = _observations_from_frame_indices(1, monitored_zone.id, [10, 11, 12])
    timeline = {1: observations}

    events = evaluate_intrusion(timeline, [monitored_zone], cfg)
    assert events == [], f"a 'monitored' zone must never produce an intrusion event, got {events}"


def test_evaluate_intrusion_ignores_unarmed_restricted_zone():
    cfg = {"intrusion_gap_frames": 2, "default_loiter_seconds": 30}
    # Armed 22:00-06:00 only; observations occur at midday (frame_timestamp
    # with fps=86400 puts frame N at N seconds after midnight -> frame 43200
    # = 12:00:00, well outside the armed window).
    zone = Zone(id=1, camera_id=1, name="Vault", zone_type="restricted",
                polygon=[], schedule={"from": "22:00", "to": "06:00"}, is_active=True)

    t0 = datetime(2026, 1, 1, 0, 0, 0)
    observations = _observations_from_frame_indices(1, zone.id, [43200, 43201, 43202], fps=1.0, t0=t0)
    timeline = {1: observations}

    events = evaluate_intrusion(timeline, [zone], cfg)
    assert events == [], f"an unarmed restricted zone must not produce an intrusion event, got {events}"


# ---------------------------------------------------------------------------
# evaluate_loitering(): fixture-driven dwell-threshold boundary
# ---------------------------------------------------------------------------

def test_evaluate_loitering_below_threshold_is_zero_events():
    fixture = _load_fixture("timeline_loitering.json")
    zones = [_zone_from_dict(z) for z in fixture["zones"]]
    case = fixture["below_threshold"]
    cfg = {"intrusion_gap_frames": fixture["gap_tolerance"], "default_loiter_seconds": 999}

    observations = _observations_from_frame_indices(
        case["track_id"], zones[0].id, case["frame_indices"], fps=fixture["fps"],
    )
    timeline = {case["track_id"]: observations}

    events = evaluate_loitering(timeline, zones, cfg)
    assert len(events) == case["expected_event_count"], \
        f"expected {case['expected_event_count']} event(s) for a 29s dwell (threshold 30s), got {len(events)}"


def test_evaluate_loitering_above_threshold_is_one_event():
    fixture = _load_fixture("timeline_loitering.json")
    zones = [_zone_from_dict(z) for z in fixture["zones"]]
    case = fixture["above_threshold"]
    cfg = {"intrusion_gap_frames": fixture["gap_tolerance"], "default_loiter_seconds": 999}

    observations = _observations_from_frame_indices(
        case["track_id"], zones[0].id, case["frame_indices"], fps=fixture["fps"],
    )
    timeline = {case["track_id"]: observations}

    events = evaluate_loitering(timeline, zones, cfg)
    assert len(events) == case["expected_event_count"], \
        f"expected {case['expected_event_count']} event(s) for a 30s dwell (threshold 30s), got {len(events)}"

    ev = events[0]
    assert ev.event_type == "loitering"
    assert ev.zone_id == zones[0].id
    assert ev.track_id == case["track_id"]
    assert abs(ev.details["dwell_seconds"] - 30.0) < 1e-9


def test_evaluate_loitering_falls_back_to_default_loiter_seconds():
    # Zone with NO schedule at all -> must fall back to
    # cfg["default_loiter_seconds"], not silently skip the zone.
    zone = Zone(id=1, camera_id=1, name="Hallway", zone_type="safe",
                polygon=[], schedule=None, is_active=True)
    cfg = {"intrusion_gap_frames": 1000, "default_loiter_seconds": 10}

    # 11s dwell >= the 10s fallback threshold -> one event.
    observations = _observations_from_frame_indices(1, zone.id, [0, 11], fps=1.0)
    timeline = {1: observations}

    events = evaluate_loitering(timeline, [zone], cfg)
    assert len(events) == 1, f"expected the default_loiter_seconds fallback to trigger one event, got {events}"


def main():
    check("timing.frame_timestamp(): basic t0 + idx/fps arithmetic", test_frame_timestamp_basic_arithmetic)
    check("timing.frame_timestamp(): matches formatter.py's prior inline formula", test_frame_timestamp_matches_formatter_style_derivation)
    check("build_tracks_timeline(): skips non-person + untracked entries", test_build_tracks_timeline_skips_non_person_and_untracked_entries)
    check("build_tracks_timeline(): frame_idx/ts derived from int(frame) via frame_timestamp()", test_build_tracks_timeline_frame_idx_and_ts_use_int_frame)
    check("build_tracks_timeline(): calib=None degrades to world_xy/zone_id=None", test_build_tracks_timeline_degrades_gracefully_without_calibration)
    check("build_tracks_timeline(): groups by track_id, frame-ordered", test_build_tracks_timeline_groups_by_track_id_frame_ordered)
    check("build_tracks_timeline(): classifies zone_id when calibrated", test_build_tracks_timeline_classifies_zone_with_calibration)
    check("SECURITY_RULES has intrusion_gap_frames + default_loiter_seconds", test_security_rules_has_intrusion_and_loitering_config_fields)
    check("evaluate_intrusion(): one-frame gap (within tolerance) -> ONE event", test_evaluate_intrusion_one_frame_gap_is_one_event)
    check("evaluate_intrusion(): gap exceeding tolerance -> TWO events", test_evaluate_intrusion_exceeding_gap_is_two_events)
    check("evaluate_intrusion(): ignores a 'monitored' (non-restricted) zone", test_evaluate_intrusion_ignores_non_restricted_zone)
    check("evaluate_intrusion(): ignores an unarmed restricted zone", test_evaluate_intrusion_ignores_unarmed_restricted_zone)
    check("evaluate_loitering(): dwell just below threshold -> zero events", test_evaluate_loitering_below_threshold_is_zero_events)
    check("evaluate_loitering(): dwell just at/above threshold -> one event", test_evaluate_loitering_above_threshold_is_one_event)
    check("evaluate_loitering(): falls back to default_loiter_seconds", test_evaluate_loitering_falls_back_to_default_loiter_seconds)

    print("\n=== Fase 4b (PR8a) zones+events contract verification ===")
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
