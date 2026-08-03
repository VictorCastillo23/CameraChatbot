"""Manual verification for the Fase 4b (PR8a) zones contract.

Not a pytest test: this repo has no test runner, so this is a plain
runnable script using bare ``assert`` statements, printing a PASS/FAIL
summary and exiting 0/1. Run it directly:

    python tests_manual/test_zones.py

What it verifies (Fase 4b / PR8a task 4b.2):

1. `security.zones.point_in_polygon()` -- ray-casting (crossing-number)
   correctness against hand-authored fixtures in
   `tests_manual/fixtures/zones_polygon.json`: inside / outside / on-edge /
   on-vertex cases for a simple square, PLUS a concave polygon (a square
   with a triangular notch) to prove the algorithm respects true polygon
   shape instead of silently falling back to a convex hull.
2. `security.zones.zone_is_armed()` -- always-armed (`schedule=None`),
   simple same-day window, day-of-week filtering, and the overnight-wrap
   case (`"from": "22:00", "to": "06:00"` crossing midnight) that a naive
   `from <= at <= to` check gets wrong.
3. `security.zones.classify_bbox_zone()` -- `None` calibration degrades to
   `None` (no crash), and the first active zone (in list/`id` order) whose
   polygon contains the bbox's world ground point wins over a
   later-in-list zone that would otherwise also match (documents the
   "overlapping zones is a config mistake, not resolved by priority logic"
   design decision).
4. `security.zones.Zone.from_row()` + `load_zones()` -- row -> dataclass
   shape, and the "no active zones for this camera" graceful-`[]` degrade
   (fake psycopg2-shaped connection/cursor, no real Postgres needed --
   same pattern as `test_fase2_geometry.py`'s `CameraCalibration.load()`
   coverage).

Uses only synthetic in-memory/JSON data throughout -- no real camera
frames, no live Postgres connection, and never touches the repo's real
`res/`, `gallery.index`/`id_map.json`/`proto_store.npy`.
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import camerachatbot.security.zones as zones_mod  # noqa: E402
from camerachatbot.security.zones import (  # noqa: E402
    Zone,
    point_in_polygon,
    zone_is_armed,
    classify_bbox_zone,
    load_zones,
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


# ---------------------------------------------------------------------------
# point_in_polygon: fixture-driven, square + concave notch
# ---------------------------------------------------------------------------

def test_point_in_polygon_square_cases():
    fixture = _load_fixture("zones_polygon.json")["square"]
    polygon = [tuple(p) for p in fixture["polygon"]]
    for case in fixture["cases"]:
        pt = tuple(case["point"])
        got = point_in_polygon(pt, polygon)
        assert got == case["expected"], \
            f"square/{case['label']}: point {pt} -> expected {case['expected']}, got {got}"


def test_point_in_polygon_concave_notch_cases():
    fixture = _load_fixture("zones_polygon.json")["concave_notch"]
    polygon = [tuple(p) for p in fixture["polygon"]]
    for case in fixture["cases"]:
        pt = tuple(case["point"])
        got = point_in_polygon(pt, polygon)
        assert got == case["expected"], \
            f"concave_notch/{case['label']}: point {pt} -> expected {case['expected']}, got {got}"

    # Explicitly re-assert the two decisive cases by name so a regression
    # that silently treats the polygon as its convex hull (i.e. floods
    # everything inside the outer square as "inside") cannot pass by
    # accident: the notch case MUST be False, the body case MUST be True,
    # and they must disagree.
    by_label = {c["label"]: tuple(c["point"]) for c in fixture["cases"]}
    inside_body = point_in_polygon(by_label["inside_main_body"], polygon)
    inside_notch = point_in_polygon(by_label["inside_notch_excluded"], polygon)
    assert inside_body is True
    assert inside_notch is False
    assert inside_body != inside_notch


def test_point_in_polygon_degenerate_polygon_returns_false():
    # Fewer than 3 vertices can never enclose a point.
    assert point_in_polygon((0, 0), []) is False
    assert point_in_polygon((0, 0), [(0, 0), (1, 1)]) is False


# ---------------------------------------------------------------------------
# zone_is_armed: schedule None / same-day window / days filter / overnight wrap
# ---------------------------------------------------------------------------

def _zone(schedule):
    return Zone(id=1, camera_id=1, name="Z", zone_type="restricted",
                polygon=[], schedule=schedule, is_active=True)


def test_zone_is_armed_none_schedule_always_armed():
    zone = _zone(None)
    assert zone_is_armed(zone, datetime(2026, 1, 1, 3, 0)) is True
    assert zone_is_armed(zone, datetime(2026, 1, 1, 15, 0)) is True


def test_zone_is_armed_same_day_window():
    zone = _zone({"from": "09:00", "to": "17:00"})
    assert zone_is_armed(zone, datetime(2026, 1, 1, 12, 0)) is True
    assert zone_is_armed(zone, datetime(2026, 1, 1, 9, 0)) is True   # inclusive lower bound
    assert zone_is_armed(zone, datetime(2026, 1, 1, 17, 0)) is True  # inclusive upper bound
    assert zone_is_armed(zone, datetime(2026, 1, 1, 8, 0)) is False
    assert zone_is_armed(zone, datetime(2026, 1, 1, 18, 0)) is False


def test_zone_is_armed_days_filter():
    # Thursday 2026-01-01 -> weekday() == 3. Only Mon(0)/Tue(1)/Wed(2) armed.
    zone = _zone({"days": [0, 1, 2], "from": "00:00", "to": "23:59"})
    assert zone_is_armed(zone, datetime(2026, 1, 1, 12, 0)) is False  # Thursday
    zone_wed = _zone({"days": [0, 1, 2], "from": "00:00", "to": "23:59"})
    assert zone_is_armed(zone_wed, datetime(2025, 12, 31, 12, 0)) is True  # Wednesday


def test_zone_is_armed_overnight_wrap():
    # 22:00 -> 06:00 crosses midnight. A naive `from <= at <= to` check
    # would report this window as NEVER armed (since 22:00 > 06:00) -- this
    # is exactly the bug the design calls out.
    zone = _zone({"from": "22:00", "to": "06:00"})
    assert zone_is_armed(zone, datetime(2026, 1, 1, 23, 0)) is True   # late night
    assert zone_is_armed(zone, datetime(2026, 1, 1, 3, 0)) is True    # past midnight
    assert zone_is_armed(zone, datetime(2026, 1, 1, 22, 0)) is True   # inclusive start
    assert zone_is_armed(zone, datetime(2026, 1, 1, 6, 0)) is True    # inclusive end
    assert zone_is_armed(zone, datetime(2026, 1, 1, 12, 0)) is False  # midday, outside window


# ---------------------------------------------------------------------------
# classify_bbox_zone: None calibration degrades, first-match-wins ordering
# ---------------------------------------------------------------------------

class _ScaleCalib:
    """Duck-types just enough of CameraCalibration for classify_bbox_zone():
    a trivial 0.01 pixel->metre scale (no rotation/translation), and the
    SAME bottom-center ground-point convention as the real class."""

    def bbox_to_world(self, bbox):
        x1, y1, x2, y2 = bbox
        gx, gy = 0.5 * (x1 + x2), float(y2)
        return (gx * 0.01, gy * 0.01)


def test_classify_bbox_zone_none_calibration_degrades_to_none():
    zone = _zone(None)
    zone.polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    result = classify_bbox_zone([0, 0, 10, 10], None, [zone])
    assert result is None


def test_classify_bbox_zone_no_match_returns_none():
    zone = _zone(None)
    zone.polygon = [(0, 0), (1, 0), (1, 1), (0, 1)]  # world (0,0)-(1,1)
    # bbox ground point in world coords: (5, 20) -- outside the tiny zone.
    result = classify_bbox_zone([0, 0, 1000, 2000], _ScaleCalib(), [zone])
    assert result is None


def test_classify_bbox_zone_first_containing_zone_wins():
    zone_a = Zone(id=1, camera_id=1, name="A", zone_type="restricted",
                  polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
                  schedule=None, is_active=True)
    zone_b = Zone(id=2, camera_id=1, name="B", zone_type="monitored",
                  polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],  # overlaps A entirely
                  schedule=None, is_active=True)

    # bbox ground point in world coords: (5.0, 5.0) -- inside BOTH zones.
    bbox = [0, 0, 1000, 1000]
    result = classify_bbox_zone(bbox, _ScaleCalib(), [zone_a, zone_b])
    assert result is not None
    assert result.id == zone_a.id, \
        f"expected the FIRST listed containing zone (id={zone_a.id}) to win, got id={result.id}"

    # Swapping list order flips which zone wins -- proves it's list order,
    # not some other priority rule (e.g. zone_type), that decides.
    result_swapped = classify_bbox_zone(bbox, _ScaleCalib(), [zone_b, zone_a])
    assert result_swapped.id == zone_b.id


def test_classify_bbox_zone_skips_inactive_zone():
    inactive = Zone(id=1, camera_id=1, name="Inactive", zone_type="restricted",
                     polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
                     schedule=None, is_active=False)
    active = Zone(id=2, camera_id=1, name="Active", zone_type="monitored",
                  polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
                  schedule=None, is_active=True)

    bbox = [0, 0, 1000, 1000]
    result = classify_bbox_zone(bbox, _ScaleCalib(), [inactive, active])
    assert result is not None
    assert result.id == active.id


# ---------------------------------------------------------------------------
# Zone.from_row() / load_zones(): fake psycopg2-shaped connection/cursor
# (same pattern as test_fase2_geometry.py's CameraCalibration.load() tests)
# ---------------------------------------------------------------------------

class _FakeCursor:
    def __init__(self, fetchall_result=None):
        self._fetchall_result = fetchall_result if fetchall_result is not None else []
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self._fetchall_result

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, fetchall_result=None):
        self._cursor = _FakeCursor(fetchall_result)
        self.closed = False

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        self.closed = True


def _with_fake_get_conn(fake_conn, fn):
    original_get_conn = zones_mod.get_conn
    zones_mod.get_conn = lambda: fake_conn
    try:
        return fn()
    finally:
        zones_mod.get_conn = original_get_conn


def test_load_zones_returns_empty_list_when_no_active_rows():
    fake_conn = _FakeConn(fetchall_result=[])
    result = _with_fake_get_conn(fake_conn, lambda: load_zones(camera_id=999))
    assert result == []
    assert fake_conn.closed is True, "load_zones() must close its connection even on the no-rows path"


def test_load_zones_happy_path_builds_zone_dataclasses():
    fake_rows = [
        (1, 7, "Vault", "restricted", [[0, 0], [10, 0], [10, 10], [0, 10]],
         {"from": "22:00", "to": "06:00", "loiter_seconds": 45}, True),
        (2, 7, "Lobby", "monitored", [[0, 0], [5, 0], [5, 5], [0, 5]], None, True),
    ]
    fake_conn = _FakeConn(fetchall_result=fake_rows)
    result = _with_fake_get_conn(fake_conn, lambda: load_zones(camera_id=7))

    assert len(result) == 2, f"expected 2 zones, got {len(result)}"
    assert all(isinstance(z, Zone) for z in result)

    z1, z2 = result
    assert z1.id == 1 and z1.name == "Vault" and z1.zone_type == "restricted"
    assert z1.polygon == [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert z1.schedule == {"from": "22:00", "to": "06:00", "loiter_seconds": 45}
    assert z1.is_active is True

    assert z2.id == 2 and z2.name == "Lobby" and z2.schedule is None
    assert fake_conn.closed is True


def main():
    check("point_in_polygon(): square inside/outside/on-edge/on-vertex", test_point_in_polygon_square_cases)
    check("point_in_polygon(): concave-notch polygon respects true shape", test_point_in_polygon_concave_notch_cases)
    check("point_in_polygon(): degenerate (<3 vertex) polygon -> False", test_point_in_polygon_degenerate_polygon_returns_false)
    check("zone_is_armed(): schedule=None -> always armed", test_zone_is_armed_none_schedule_always_armed)
    check("zone_is_armed(): same-day window, inclusive bounds", test_zone_is_armed_same_day_window)
    check("zone_is_armed(): days-of-week filter", test_zone_is_armed_days_filter)
    check("zone_is_armed(): overnight wrap (22:00 -> 06:00)", test_zone_is_armed_overnight_wrap)
    check("classify_bbox_zone(): calib=None -> None", test_classify_bbox_zone_none_calibration_degrades_to_none)
    check("classify_bbox_zone(): no containing zone -> None", test_classify_bbox_zone_no_match_returns_none)
    check("classify_bbox_zone(): first containing zone in list order wins", test_classify_bbox_zone_first_containing_zone_wins)
    check("classify_bbox_zone(): inactive zone skipped", test_classify_bbox_zone_skips_inactive_zone)
    check("load_zones(): [] on no active rows", test_load_zones_returns_empty_list_when_no_active_rows)
    check("load_zones(): happy path builds Zone dataclasses", test_load_zones_happy_path_builds_zone_dataclasses)

    print("\n=== Fase 4b (PR8a) zones contract verification ===")
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
