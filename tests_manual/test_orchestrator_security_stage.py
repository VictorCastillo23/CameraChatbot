"""Manual verification for the Fase 4b (PR8b) orchestrator-wiring contract.

Not a pytest test: this repo has no test runner, so this is a plain
runnable script using bare ``assert`` statements, printing a PASS/FAIL
summary and exiting 0/1. Run it directly:

    python tests_manual/test_orchestrator_security_stage.py

`zones.py`/`events.py` (PR8a) already have their own dedicated test files
and are NOT re-tested here. This file covers only the NEW logic PR8b adds:
the orchestrator-level driver that wires those two modules into
`orchestrator.multi_models()`, plus the two small serialization helpers
(`security.events.event_to_dict`, `db.postgres_writer.prepare_event_rows`/
`_event_keyframe_id`) and `video_schema.formatter`'s new `events` ride-along
node -- all pure/DB-free logic, exercised with synthetic in-memory data.

What it verifies:

1. `orchestrator._write_zone_and_world_xy()` -- writes `zone_id`=`None`/
   `world_xy`=`None` onto every person entry with `calib=None` (the
   documented degrade path); classifies the correct zone/world point with a
   fake calibration + covering zone; leaves non-person entries untouched.
2. `orchestrator._run_zones_events_stage()` -- with no `camera_id` at all
   (today's actual default from every real call site, see apply-progress):
   degrades to zero calibration/zones lookups, zero events, and still dumps
   an empty `security_events.json` checkpoint. With a `camera_id` AND a
   monkeypatched `CameraCalibration`/`load_zones` (module-level names
   `orchestrator.py` binds via `from ... import ...`, patched the same way
   `test_zones.py`/`test_fase2_geometry.py` patch `get_conn`): builds a real
   intrusion event from an always-armed restricted zone and writes
   `zone_id`/`world_xy` back onto every person entry. Also verifies a
   `CameraCalibration.load()`/`load_zones()` exception (e.g. DB
   unreachable) degrades the same way instead of propagating and aborting
   the whole pipeline run.

   Fase 5 (PR9): `AuthorizationRegistry.load()` (also a module-level name
   `orchestrator.py` binds via `from ... import ...`) now runs
   UNCONDITIONALLY inside this stage, regardless of `camera_id` -- every
   test below therefore ALSO monkeypatches `orchestrator_mod.
   AuthorizationRegistry` with a synthetic, DB-free fake (see
   `_fake_auth_registry_cls()`), preserving this file's own "no live
   Postgres connection" invariant. Also verifies an `AuthorizationRegistry.
   load()` failure degrades to `registry=None` (skipping `evaluate_
   unenrolled` for this run) rather than raising, and that a real
   `unenrolled_person` event IS produced end-to-end when a tracked person's
   `person_global_id` is not in the (fake) registry for long enough to
   clear `SECURITY_RULES["unenrolled_debounce_frames"]`.
3. `security.events.event_to_dict()` -- ISO-8601 `started_at`, `None`
   `ended_at` stays `None` (not `"None"` the string), all other fields pass
   through verbatim.
4. `video_schema.formatter.reformat_to_video_schema_uniform()` -- the new
   `events` kwarg attaches a `video.events` node (list of `event_to_dict()`
   output), the same ride-along position as the pre-existing `neighborhood`
   node.
5. `db.postgres_writer._event_keyframe_id()` -- bounds-checked
   `keyframe_ids[event["first_frame_idx"]]`, returns `None` instead of
   raising `IndexError` on an out-of-range/missing index.
6. `db.postgres_writer.prepare_event_rows()` -- shapes each event dict into
   the exact positional tuple `insert_events_in_batches()`'s `execute_values`
   template expects, including wrapping `details` in `psycopg2.extras.Json`.

Uses only synthetic in-memory data throughout -- no real camera frames, no
ONNX models, no live Postgres connection, and never touches the repo's real
`keyFrames/`, `res/`, or `gallery.index`/`id_map.json`/`proto_store.npy`.
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psycopg2.extras import Json  # noqa: E402

import camerachatbot.pipeline.orchestrator as orchestrator_mod  # noqa: E402
from camerachatbot.pipeline.orchestrator import (  # noqa: E402
    _write_zone_and_world_xy,
    _run_zones_events_stage,
)
from camerachatbot.security.events import SecurityEvent, event_to_dict  # noqa: E402
from camerachatbot.security.zones import Zone  # noqa: E402
from camerachatbot.video_schema.formatter import reformat_to_video_schema_uniform  # noqa: E402
from camerachatbot.db.postgres_writer import prepare_event_rows, _event_keyframe_id  # noqa: E402

results = []  # list[tuple[str, bool, str]]


def check(name, fn):
    try:
        fn()
        results.append((name, True, ""))
    except AssertionError as e:
        results.append((name, False, f"AssertionError: {e}"))
    except Exception as e:  # noqa: BLE001 - report any unexpected exception as a failure
        results.append((name, False, f"{type(e).__name__}: {e}"))


class _FakeReID:
    """Minimal stand-in for YOLOPersonReID -- just the three attributes the
    PR8b zones+events stage reads/writes: `frame_order`, `results_json`
    (mutated in place), and `output_folder` (where the checkpoint lands)."""

    def __init__(self, frame_order, results_json, output_folder):
        self.frame_order = frame_order
        self.results_json = results_json
        self.output_folder = output_folder


class _FakeCalib:
    """Duck-types just enough of CameraCalibration: a trivial 0.01
    pixel->metre scale, bottom-center ground point (same convention as
    `test_events.py`'s `_FakeCalib`)."""

    def bbox_to_world(self, bbox):
        x1, y1, x2, y2 = bbox
        return (0.5 * (x1 + x2) * 0.01, float(y2) * 0.01)


def _with_patched_orchestrator(monkeys, fn):
    originals = {name: getattr(orchestrator_mod, name) for name in monkeys}
    for name, val in monkeys.items():
        setattr(orchestrator_mod, name, val)
    try:
        return fn()
    finally:
        for name, val in originals.items():
            setattr(orchestrator_mod, name, val)


def _covering_zone(zone_id=9, zone_type="restricted"):
    # World (0,0)-(100,100), covers any ground point from the small pixel
    # bboxes used below (scaled by 0.01 via _FakeCalib).
    return Zone(id=zone_id, camera_id=5, name="Vault", zone_type=zone_type,
                polygon=[(0, 0), (100, 0), (100, 100), (0, 100)], schedule=None, is_active=True)


class _FakeAuthRegistry:
    """Duck-types just enough of `identity.authorization.
    AuthorizationRegistry` for these tests: a plain set of authorized
    person_global_ids, no real DB."""

    def __init__(self, authorized_pids=()):
        self._authorized = set(authorized_pids)

    def is_authorized(self, person_global_id):
        return person_global_id is not None and person_global_id in self._authorized


def _fake_auth_registry_cls(authorized_pids=()):
    """A module-patchable class exposing the same `load()` classmethod
    contract as the real `AuthorizationRegistry` -- avoids a live Postgres
    connection attempt from `AuthorizationRegistry.load()` (Fase 5, PR9),
    which now runs unconditionally inside `_run_zones_events_stage()`
    regardless of `camera_id`."""

    class _Cls:
        @classmethod
        def load(cls):
            return _FakeAuthRegistry(authorized_pids)

    return _Cls


# ---------------------------------------------------------------------------
# orchestrator._write_zone_and_world_xy()
# ---------------------------------------------------------------------------

def test_write_zone_and_world_xy_degrades_with_no_calibration():
    results_json = {
        "0": [
            {"kind": "person", "bbox": [0, 0, 10, 20]},
            {"kind": "object", "bbox": [5, 5, 15, 15]},
        ],
    }
    reid = _FakeReID(["0"], results_json, output_folder=None)
    _write_zone_and_world_xy(reid, calib=None, zones=[])

    person = results_json["0"][0]
    assert person["zone_id"] is None
    assert person["world_xy"] is None
    obj = results_json["0"][1]
    assert "zone_id" not in obj, "non-person entries must not be touched"


def test_write_zone_and_world_xy_classifies_with_calibration():
    zone = _covering_zone()
    results_json = {"0": [{"kind": "person", "bbox": [0, 0, 10, 20]}]}
    reid = _FakeReID(["0"], results_json, output_folder=None)
    _write_zone_and_world_xy(reid, calib=_FakeCalib(), zones=[zone])

    person = results_json["0"][0]
    assert person["zone_id"] == 9
    assert person["world_xy"] == [0.05, 0.2]


# ---------------------------------------------------------------------------
# orchestrator._run_zones_events_stage()
# ---------------------------------------------------------------------------

def test_run_zones_events_stage_no_camera_id_degrades_and_dumps_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        results_json = {
            "0": [{"kind": "person", "bbox": [0, 0, 10, 20], "track_id": 1,
                    "person_global_id": 7, "confidence": 0.9}],
        }
        reid = _FakeReID(["0"], results_json, output_folder=tmpdir)

        # Only 1 observation -- well below SECURITY_RULES["unenrolled_
        # debounce_frames"] (5), so an empty (authorizes-nothing) fake
        # registry still contributes zero events here.
        events = _with_patched_orchestrator(
            {"AuthorizationRegistry": _fake_auth_registry_cls()},
            lambda: _run_zones_events_stage(reid, camera_id=None, start_at="2026-01-01T00:00:00Z", fps=10),
        )

        assert events == []
        person = results_json["0"][0]
        assert person["zone_id"] is None
        assert person["world_xy"] is None

        ckpt_path = os.path.join(tmpdir, "security_events.json")
        assert os.path.exists(ckpt_path)
        with open(ckpt_path, "r", encoding="utf-8") as f:
            assert json.load(f) == []


def test_run_zones_events_stage_with_camera_id_builds_intrusion_event():
    zone = _covering_zone()

    class _FakeCalibCls:
        @classmethod
        def load(cls, camera_id):
            assert camera_id == 5
            return _FakeCalib()

    def _fake_load_zones(camera_id):
        assert camera_id == 5
        return [zone]

    with tempfile.TemporaryDirectory() as tmpdir:
        results_json = {
            str(i): [{"kind": "person", "bbox": [0, 0, 10, 20], "track_id": 1,
                       "person_global_id": 7, "confidence": 0.9}]
            for i in range(3)
        }
        reid = _FakeReID([str(i) for i in range(3)], results_json, output_folder=tmpdir)

        # 3 observations -- still below the debounce threshold (5), so an
        # empty fake registry contributes zero unenrolled_person events;
        # only the intrusion event is expected.
        events = _with_patched_orchestrator(
            {"CameraCalibration": _FakeCalibCls, "load_zones": _fake_load_zones,
             "AuthorizationRegistry": _fake_auth_registry_cls()},
            lambda: _run_zones_events_stage(reid, camera_id=5, start_at="2026-01-01T00:00:00Z", fps=10),
        )

        assert len(events) == 1, f"expected exactly one intrusion event, got {len(events)}"
        ev = events[0]
        assert ev.event_type == "intrusion"
        assert ev.zone_id == 9
        assert ev.track_id == 1
        assert ev.person_global_id == 7

        for i in range(3):
            person = results_json[str(i)][0]
            assert person["zone_id"] == 9
            assert person["world_xy"] == [0.05, 0.2]

        ckpt_path = os.path.join(tmpdir, "security_events.json")
        with open(ckpt_path, "r", encoding="utf-8") as f:
            dumped = json.load(f)
        assert len(dumped) == 1
        assert dumped[0]["event_type"] == "intrusion"


def test_run_zones_events_stage_degrades_when_calibration_lookup_raises():
    class _RaisingCalibCls:
        @classmethod
        def load(cls, camera_id):
            raise RuntimeError("db unreachable")

    def _raising_load_zones(camera_id):
        raise RuntimeError("db unreachable")

    with tempfile.TemporaryDirectory() as tmpdir:
        results_json = {
            "0": [{"kind": "person", "bbox": [0, 0, 10, 20], "track_id": 1,
                    "person_global_id": 7, "confidence": 0.9}],
        }
        reid = _FakeReID(["0"], results_json, output_folder=tmpdir)

        events = _with_patched_orchestrator(
            {"CameraCalibration": _RaisingCalibCls, "load_zones": _raising_load_zones,
             "AuthorizationRegistry": _fake_auth_registry_cls()},
            lambda: _run_zones_events_stage(reid, camera_id=5, start_at="2026-01-01T00:00:00Z", fps=10),
        )

        assert events == [], "a DB/lookup failure must degrade to no events, not raise"
        person = results_json["0"][0]
        assert person["zone_id"] is None
        assert person["world_xy"] is None


def test_run_zones_events_stage_degrades_on_malformed_zone_schedule():
    """Gate-review fix regression test: a zone with a malformed
    `schedule["from"]`/`["to"]` string (e.g. `"22h00"` instead of `"22:00"`)
    used to crash `zones.py::_parse_hhmm()` with
    `ValueError: not enough values to unpack`, and that propagated all the
    way up through `evaluate_intrusion()` and out of
    `_run_zones_events_stage()`, aborting the whole `multi_models()` call
    (pose classification + Postgres persistence included) for the entire
    batch -- not just the security-events stage. Only the calib/zones DB
    LOOKUPS were try/excepted before this fix; everything downstream
    (`_write_zone_and_world_xy`, `build_tracks_timeline`,
    `evaluate_intrusion`, `evaluate_loitering`) was not.
    """
    bad_zone = Zone(
        id=9, camera_id=5, name="Vault", zone_type="restricted",
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
        schedule={"from": "22h00", "to": "06:00"},  # malformed -- no ":" split
        is_active=True,
    )

    class _FakeCalibCls:
        @classmethod
        def load(cls, camera_id):
            return _FakeCalib()

    def _fake_load_zones(camera_id):
        return [bad_zone]

    with tempfile.TemporaryDirectory() as tmpdir:
        results_json = {
            "0": [{"kind": "person", "bbox": [0, 0, 10, 20], "track_id": 1,
                    "person_global_id": 7, "confidence": 0.9}],
        }
        reid = _FakeReID(["0"], results_json, output_folder=tmpdir)

        events = _with_patched_orchestrator(
            {"CameraCalibration": _FakeCalibCls, "load_zones": _fake_load_zones,
             "AuthorizationRegistry": _fake_auth_registry_cls()},
            lambda: _run_zones_events_stage(reid, camera_id=5, start_at="2026-01-01T00:00:00Z", fps=10),
        )

        assert events == [], "a malformed zone schedule must degrade to no events, not crash the batch"

        ckpt_path = os.path.join(tmpdir, "security_events.json")
        assert os.path.exists(ckpt_path), "the events checkpoint must still be dumped on degrade"
        with open(ckpt_path, "r", encoding="utf-8") as f:
            assert json.load(f) == []


def test_run_zones_events_stage_degrades_on_fps_zero():
    """Gate-review fix regression test: `fps=0` used to crash with
    `ZeroDivisionError` inside `video_schema.timing.frame_timestamp()`
    (called from `build_tracks_timeline()` for EVERY observation,
    regardless of whether a camera is calibrated), propagating out of
    `_run_zones_events_stage()` the same way the malformed-schedule case
    did."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_json = {
            "0": [{"kind": "person", "bbox": [0, 0, 10, 20], "track_id": 1,
                    "person_global_id": 7, "confidence": 0.9}],
        }
        reid = _FakeReID(["0"], results_json, output_folder=tmpdir)

        # No camera_id needed: build_tracks_timeline() computes
        # frame_timestamp() for every observation unconditionally, so
        # fps=0 crashes even with calib=None/zones=[].
        events = _with_patched_orchestrator(
            {"AuthorizationRegistry": _fake_auth_registry_cls()},
            lambda: _run_zones_events_stage(reid, camera_id=None, start_at="2026-01-01T00:00:00Z", fps=0),
        )

        assert events == [], "fps=0 must degrade to no events, not crash the batch"

        ckpt_path = os.path.join(tmpdir, "security_events.json")
        assert os.path.exists(ckpt_path), "the events checkpoint must still be dumped on degrade"
        with open(ckpt_path, "r", encoding="utf-8") as f:
            assert json.load(f) == []


# ---------------------------------------------------------------------------
# orchestrator._run_zones_events_stage(): Fase 5 (PR9) AuthorizationRegistry
# wiring -- load-failure degrade + real evaluate_unenrolled wiring
# ---------------------------------------------------------------------------

def test_run_zones_events_stage_degrades_when_authorization_registry_load_raises():
    """Fase 5 (PR9) gate-review-style regression coverage: an
    `AuthorizationRegistry.load()` failure (e.g. DB unreachable) must
    degrade to `registry=None` (skipping `evaluate_unenrolled` entirely for
    this run) rather than raising out of `_run_zones_events_stage()` --
    mirroring the `CameraCalibration`/`load_zones` degrade convention
    already established (task requirement: "orchestrator wiring degrade
    test ... mirroring the existing calibration/zone degrade tests")."""

    class _RaisingAuthCls:
        @classmethod
        def load(cls):
            raise RuntimeError("db unreachable")

    with tempfile.TemporaryDirectory() as tmpdir:
        results_json = {
            "0": [{"kind": "person", "bbox": [0, 0, 10, 20], "track_id": 1,
                    "person_global_id": 7, "confidence": 0.9}],
        }
        reid = _FakeReID(["0"], results_json, output_folder=tmpdir)

        events = _with_patched_orchestrator(
            {"AuthorizationRegistry": _RaisingAuthCls},
            lambda: _run_zones_events_stage(reid, camera_id=None, start_at="2026-01-01T00:00:00Z", fps=10),
        )

        assert events == [], "an AuthorizationRegistry load failure must degrade to no events, not raise"

        ckpt_path = os.path.join(tmpdir, "security_events.json")
        assert os.path.exists(ckpt_path), "the events checkpoint must still be dumped on degrade"
        with open(ckpt_path, "r", encoding="utf-8") as f:
            assert json.load(f) == []


def test_run_zones_events_stage_wires_evaluate_unenrolled_end_to_end():
    """Fase 5 (PR9): with a real (fake) `AuthorizationRegistry` and enough
    consecutive unauthorized observations to clear `SECURITY_RULES[
    "unenrolled_debounce_frames"]`, `_run_zones_events_stage()` must include
    an `unenrolled_person` event in its returned list, proving `evaluate_
    unenrolled()` is actually wired in (not just importable)."""
    from camerachatbot.security_config import SECURITY_RULES
    debounce = SECURITY_RULES["unenrolled_debounce_frames"]

    with tempfile.TemporaryDirectory() as tmpdir:
        frames = [str(i) for i in range(debounce)]
        results_json = {
            f: [{"kind": "person", "bbox": [0, 0, 10, 20], "track_id": 1,
                 "person_global_id": 99, "confidence": 0.9}]
            for f in frames
        }
        reid = _FakeReID(frames, results_json, output_folder=tmpdir)

        # pid 99 is NOT in the fake registry -> unauthorized for
        # `debounce` consecutive frames -> exactly one unenrolled_person
        # event expected, with no camera_id/calibration/zones involved.
        events = _with_patched_orchestrator(
            {"AuthorizationRegistry": _fake_auth_registry_cls(authorized_pids=())},
            lambda: _run_zones_events_stage(reid, camera_id=None, start_at="2026-01-01T00:00:00Z", fps=10),
        )

        unenrolled = [e for e in events if e.event_type == "unenrolled_person"]
        assert len(unenrolled) == 1, f"expected one unenrolled_person event, got {events}"
        assert unenrolled[0].person_global_id == 99
        assert unenrolled[0].track_id == 1

        ckpt_path = os.path.join(tmpdir, "security_events.json")
        with open(ckpt_path, "r", encoding="utf-8") as f:
            dumped = json.load(f)
        assert len(dumped) == 1
        assert dumped[0]["event_type"] == "unenrolled_person"


def test_run_zones_events_stage_authorized_person_produces_no_unenrolled_event():
    """Same setup as the wiring test above, but pid 99 IS in the fake
    registry -- must produce zero `unenrolled_person` events even though
    the run clears the debounce threshold."""
    from camerachatbot.security_config import SECURITY_RULES
    debounce = SECURITY_RULES["unenrolled_debounce_frames"]

    with tempfile.TemporaryDirectory() as tmpdir:
        frames = [str(i) for i in range(debounce)]
        results_json = {
            f: [{"kind": "person", "bbox": [0, 0, 10, 20], "track_id": 1,
                 "person_global_id": 99, "confidence": 0.9}]
            for f in frames
        }
        reid = _FakeReID(frames, results_json, output_folder=tmpdir)

        events = _with_patched_orchestrator(
            {"AuthorizationRegistry": _fake_auth_registry_cls(authorized_pids=(99,))},
            lambda: _run_zones_events_stage(reid, camera_id=None, start_at="2026-01-01T00:00:00Z", fps=10),
        )

        unenrolled = [e for e in events if e.event_type == "unenrolled_person"]
        assert unenrolled == [], f"an authorized person must not produce an unenrolled_person event, got {events}"


# ---------------------------------------------------------------------------
# security.events.event_to_dict()
# ---------------------------------------------------------------------------

def test_event_to_dict_serializes_timestamps_and_none_ended_at():
    ev = SecurityEvent(
        event_type="loitering", zone_id=2, person_global_id=9, track_id=4,
        started_at=datetime(2026, 1, 1, 12, 0, 0), ended_at=None,
        first_frame_idx=3, confidence=0.75, details={"dwell_seconds": 12.0},
    )
    d = event_to_dict(ev)
    assert d["event_type"] == "loitering"
    assert d["zone_id"] == 2
    assert d["started_at"] == "2026-01-01T12:00:00"
    assert d["ended_at"] is None, "None ended_at must stay None, not become the string 'None'"
    assert d["first_frame_idx"] == 3
    assert d["confidence"] == 0.75
    assert d["details"] == {"dwell_seconds": 12.0}


# ---------------------------------------------------------------------------
# video_schema.formatter: video.events ride-along node
# ---------------------------------------------------------------------------

def test_formatter_attaches_events_node():
    ev = SecurityEvent(
        event_type="intrusion", zone_id=1, person_global_id=2, track_id=3,
        started_at=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
        first_frame_idx=0, confidence=None, details={"zone_name": "Vault"},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "src.json")
        with open(src_path, "w", encoding="utf-8") as f:
            json.dump({"0": [], "neighborhood": []}, f)
        dst_path = os.path.join(tmpdir, "dst.json")

        reformat_to_video_schema_uniform(
            src_path, dst_path,
            video_key="k", start_at="2026-01-01T00:00:00Z", size_xy=(10, 10), fps=1,
            per_frame_inference=0.0, per_frame_preprocess=0.0, per_frame_postprocess=0.0,
            events=[ev],
        )

        with open(dst_path, "r", encoding="utf-8") as f:
            out = json.load(f)

    assert "events" in out["video"], "video.events node missing"
    assert len(out["video"]["events"]) == 1
    got = out["video"]["events"][0]
    assert got["event_type"] == "intrusion"
    assert got["zone_id"] == 1
    assert got["started_at"] == "2026-01-01T00:00:00+00:00"


def test_formatter_events_defaults_to_empty_list_when_omitted():
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "src.json")
        with open(src_path, "w", encoding="utf-8") as f:
            json.dump({"0": [], "neighborhood": []}, f)
        dst_path = os.path.join(tmpdir, "dst.json")

        reformat_to_video_schema_uniform(
            src_path, dst_path,
            video_key="k", start_at="2026-01-01T00:00:00Z", size_xy=(10, 10), fps=1,
            per_frame_inference=0.0, per_frame_preprocess=0.0, per_frame_postprocess=0.0,
        )

        with open(dst_path, "r", encoding="utf-8") as f:
            out = json.load(f)

    assert out["video"]["events"] == []


# ---------------------------------------------------------------------------
# db.postgres_writer: _event_keyframe_id() / prepare_event_rows()
# ---------------------------------------------------------------------------

def test_event_keyframe_id_bounds_checked():
    keyframe_ids = [101, 102, 103]
    assert _event_keyframe_id({"first_frame_idx": 1}, keyframe_ids) == 102
    assert _event_keyframe_id({"first_frame_idx": 99}, keyframe_ids) is None
    assert _event_keyframe_id({"first_frame_idx": -1}, keyframe_ids) is None
    assert _event_keyframe_id({"first_frame_idx": None}, keyframe_ids) is None
    assert _event_keyframe_id({}, keyframe_ids) is None


def test_prepare_event_rows_shapes_tuple_for_execute_values():
    keyframe_ids = [55, 56]
    events = [{
        "event_type": "intrusion", "zone_id": 3, "person_global_id": 9, "track_id": 4,
        "started_at": "2026-01-01T00:00:00", "ended_at": "2026-01-01T00:00:05",
        "first_frame_idx": 1, "confidence": None, "details": {"zone_name": "Vault"},
    }]
    rows = prepare_event_rows(video_id=10, camera_id=2, events=events, keyframe_ids=keyframe_ids)

    assert len(rows) == 1
    row = rows[0]
    assert row[0] == 10, "video_id"
    assert row[1] == 2, "camera_id"
    assert row[2] == 3, "zone_id"
    assert row[3] == 56, "key_frame_id == keyframe_ids[first_frame_idx]"
    assert row[4] == "intrusion", "event_type"
    assert row[5] == 9, "person_global_id"
    assert row[6] == 4, "track_id"
    assert row[7] == "2026-01-01T00:00:00", "started_at"
    assert row[8] == "2026-01-01T00:00:05", "ended_at"
    assert row[9] is None, "confidence"
    assert isinstance(row[10], Json), "details must be wrapped in psycopg2.extras.Json"


def main():
    check("_write_zone_and_world_xy(): calib=None degrades to None/None",
          test_write_zone_and_world_xy_degrades_with_no_calibration)
    check("_write_zone_and_world_xy(): classifies zone/world_xy with a calibration",
          test_write_zone_and_world_xy_classifies_with_calibration)

    check("_run_zones_events_stage(): no camera_id -> degrades, dumps empty checkpoint",
          test_run_zones_events_stage_no_camera_id_degrades_and_dumps_checkpoint)
    check("_run_zones_events_stage(): camera_id + calib/zones -> real intrusion event",
          test_run_zones_events_stage_with_camera_id_builds_intrusion_event)
    check("_run_zones_events_stage(): calibration/zone lookup failure degrades, does not raise",
          test_run_zones_events_stage_degrades_when_calibration_lookup_raises)
    check("_run_zones_events_stage(): malformed zone schedule degrades, does not crash the batch",
          test_run_zones_events_stage_degrades_on_malformed_zone_schedule)
    check("_run_zones_events_stage(): fps=0 degrades, does not crash the batch",
          test_run_zones_events_stage_degrades_on_fps_zero)

    check("_run_zones_events_stage(): AuthorizationRegistry load failure degrades, does not raise",
          test_run_zones_events_stage_degrades_when_authorization_registry_load_raises)
    check("_run_zones_events_stage(): wires evaluate_unenrolled() end-to-end",
          test_run_zones_events_stage_wires_evaluate_unenrolled_end_to_end)
    check("_run_zones_events_stage(): authorized person produces no unenrolled_person event",
          test_run_zones_events_stage_authorized_person_produces_no_unenrolled_event)

    check("event_to_dict(): ISO timestamps, None ended_at stays None",
          test_event_to_dict_serializes_timestamps_and_none_ended_at)

    check("formatter: video.events node carries event_to_dict() output",
          test_formatter_attaches_events_node)
    check("formatter: video.events defaults to [] when events= omitted",
          test_formatter_events_defaults_to_empty_list_when_omitted)

    check("_event_keyframe_id(): bounds-checked, None on out-of-range/missing",
          test_event_keyframe_id_bounds_checked)
    check("prepare_event_rows(): tuple shape + Json-wrapped details",
          test_prepare_event_rows_shapes_tuple_for_execute_values)

    print("\n=== Fase 4b (PR8b) + Fase 5 (PR9) orchestrator-wiring contract verification ===")
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
