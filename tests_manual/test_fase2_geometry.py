"""Manual verification for the Fase 2 (PR3) geometry contract.

Not a pytest test: this repo has no test runner, so this is a plain
runnable script using bare ``assert`` statements, printing a PASS/FAIL
summary and exiting 0/1. Run it directly:

    python tests_manual/test_fase2_geometry.py

What it verifies (Fase 2 / PR3 tasks):

1. `geometry.bbox_utils.center()` / `iou()` / `bbox_area()` / `clamp_bbox()`
   — pure math, matches the closures/inline logic they were extracted from
   in `detectors/person_reid.py`.
2. `geometry.homography.CameraCalibration.bbox_ground_point()` returns the
   bbox BOTTOM-CENTER, not the center — the whole reason the design calls
   this out explicitly (a homography only maps the ground plane; feet, not
   torso-center, are the reliably-on-plane point).
3. `geometry.homography.CameraCalibration.image_to_world()` /
   `bbox_to_world()` correctly delegate to `cv2.perspectiveTransform()`
   against a known, hand-computed homography (a simple pixel->metre scale),
   so the wiring (reshape in/out, dtype) is exercised without needing a real
   calibrated camera.
4. `geometry.homography.CameraCalibration.load()` gracefully returns `None`
   when the `camera_calibration` table has no active row for the given
   camera — exercised with a fake psycopg2-shaped connection/cursor
   (monkeypatched `get_conn`), so no real Postgres connection is needed.
   Also verifies the happy path: a fake row makes `load()` return a
   correctly-shaped `CameraCalibration` via `from_row()`.
5. `geometry.calibrate_camera._compute_homography()` — pure function, no
   OpenCV window/display needed — recovers a known homography (within
   floating-point tolerance) from synthetic image/world point
   correspondences, and reports ~0 reprojection error for a noiseless input.

Uses only synthetic in-memory data throughout — no real camera, no display,
no live Postgres connection, and never touches the repo's real `models/`,
`gallery.index`/`id_map.json`/`proto_store.npy` (see the "Identity
persistence gotcha" note in CLAUDE.md).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from camerachatbot.geometry.bbox_utils import center, iou, bbox_area, clamp_bbox  # noqa: E402
import camerachatbot.geometry.homography as homography_mod  # noqa: E402
from camerachatbot.geometry.homography import CameraCalibration  # noqa: E402
from camerachatbot.geometry.calibrate_camera import _compute_homography  # noqa: E402

results = []  # list[tuple[str, bool, str]]


def check(name, fn):
    try:
        fn()
        results.append((name, True, ""))
    except AssertionError as e:
        results.append((name, False, f"AssertionError: {e}"))
    except Exception as e:  # noqa: BLE001 - report any unexpected exception as a failure
        results.append((name, False, f"{type(e).__name__}: {e}"))


# ---------------------------------------------------------------------------
# bbox_utils: center / iou / bbox_area / clamp_bbox
# ---------------------------------------------------------------------------

def test_center():
    assert center([0, 0, 10, 20]) == (5.0, 10.0)
    assert center([10, 10, 30, 50]) == (20.0, 30.0)


def test_iou_known_cases():
    # No overlap.
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    # Identical boxes -> IoU 1.0.
    assert abs(iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-9
    # Half-overlap: [0,0,10,10] vs [5,0,15,10] -> intersection 5x10=50,
    # union 100+100-50=150 -> IoU = 1/3.
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - (1.0 / 3.0)) < 1e-9


def test_bbox_area():
    assert bbox_area([0, 0, 10, 20]) == 200.0
    # Degenerate/negative-size bbox -> area 0, not negative.
    assert bbox_area([10, 10, 5, 5]) == 0.0


def test_clamp_bbox_clips_out_of_bounds():
    # Box partially outside a 100x50 image on every side.
    x1, y1, x2, y2 = clamp_bbox([-5, -5, 150, 80], W=100, H=50)
    assert (x1, y1, x2, y2) == (0, 0, 99, 49), (x1, y1, x2, y2)

    # In-bounds box passes through unchanged (already-int coordinates).
    x1, y1, x2, y2 = clamp_bbox([10, 10, 20, 20], W=100, H=50)
    assert (x1, y1, x2, y2) == (10, 10, 20, 20)

    # Rounds to nearest int before clamping.
    x1, y1, x2, y2 = clamp_bbox([1.4, 1.6, 10.5, 10.4], W=100, H=50)
    assert (x1, y1, x2, y2) == (1, 2, 10, 10), (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# homography: bbox_ground_point is bottom-center, NOT bbox center
# ---------------------------------------------------------------------------

def test_bbox_ground_point_is_bottom_center_not_center():
    bbox = [0, 0, 10, 20]
    ground = CameraCalibration.bbox_ground_point(bbox)
    bbox_center = center(bbox)

    assert ground == (5.0, 20.0), ground
    assert ground != bbox_center, \
        f"bbox_ground_point must NOT equal bbox center — got {ground} == {bbox_center}"
    # The y-coordinate must be the bbox bottom (y2), not the vertical midpoint.
    assert ground[1] == bbox[3]


# ---------------------------------------------------------------------------
# homography: image_to_world / bbox_to_world wiring against a known H
# ---------------------------------------------------------------------------

def _scale_calibration(scale=0.01, camera_id=1):
    """A trivial, hand-computed homography: image pixels -> world metres via
    a uniform scale (no rotation/translation), so expected outputs can be
    checked by hand instead of round-tripping through cv2.findHomography."""
    H = np.array([
        [scale, 0.0, 0.0],
        [0.0, scale, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return CameraCalibration(
        camera_id=camera_id, H=H, units="m", reference_points=[], reprojection_error=0.0,
    )


def test_image_to_world_known_scale():
    calib = _scale_calibration(scale=0.01)
    world = calib.image_to_world(np.array([[100.0, 200.0], [0.0, 0.0]]))
    assert world.shape == (2, 2)
    assert abs(world[0, 0] - 1.0) < 1e-9 and abs(world[0, 1] - 2.0) < 1e-9
    assert abs(world[1, 0] - 0.0) < 1e-9 and abs(world[1, 1] - 0.0) < 1e-9


def test_bbox_to_world_uses_ground_point():
    calib = _scale_calibration(scale=0.01)
    bbox = [0, 0, 100, 200]  # ground point (bottom-center) = (50, 200)
    wx, wy = calib.bbox_to_world(bbox)
    assert abs(wx - 0.5) < 1e-9 and abs(wy - 2.0) < 1e-9, (wx, wy)


# ---------------------------------------------------------------------------
# homography: CameraCalibration.load() graceful None / happy path
# (fake psycopg2-shaped connection — no real DB needed)
# ---------------------------------------------------------------------------

class _FakeCursor:
    def __init__(self, fetchone_result=None):
        self._fetchone_result = fetchone_result
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self._fetchone_result

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, fetchone_result=None):
        self._cursor = _FakeCursor(fetchone_result)
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
    original_get_conn = homography_mod.get_conn
    homography_mod.get_conn = lambda: fake_conn
    try:
        return fn()
    finally:
        homography_mod.get_conn = original_get_conn


def test_camera_calibration_load_returns_none_when_no_row():
    fake_conn = _FakeConn(fetchone_result=None)
    result = _with_fake_get_conn(fake_conn, lambda: CameraCalibration.load(camera_id=999))
    assert result is None
    assert fake_conn.closed is True, "load() must close its connection even on the no-row path"


def test_camera_calibration_load_happy_path():
    fake_row = (7, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], "m",
                [{"image_xy": [0, 0], "world_xy": [0, 0]}], 0.42)
    fake_conn = _FakeConn(fetchone_result=fake_row)
    result = _with_fake_get_conn(fake_conn, lambda: CameraCalibration.load(camera_id=7))

    assert result is not None
    assert result.camera_id == 7
    assert result.H.shape == (3, 3)
    assert np.allclose(result.H, np.eye(3))
    assert result.units == "m"
    assert result.reprojection_error == 0.42
    assert fake_conn.closed is True


# ---------------------------------------------------------------------------
# calibrate_camera: _compute_homography() pure math (no display, no DB)
# ---------------------------------------------------------------------------

def test_compute_homography_recovers_known_scale():
    # 4 correspondences for a pure 0.02 pixel->metre scale (a square in image
    # space maps to a smaller square in world space, no rotation).
    scale = 0.02
    image_pts = [[0, 0], [100, 0], [100, 100], [0, 100]]
    points = [
        {"image_xy": [float(x), float(y)], "world_xy": [float(x) * scale, float(y) * scale]}
        for x, y in image_pts
    ]

    H, reproj_error = _compute_homography(points)

    assert H.shape == (3, 3)
    assert reproj_error < 1e-6, f"expected ~0 reprojection error for noiseless input, got {reproj_error}"

    # Recovered homography should map a held-out point the same way the
    # known scale would.
    test_pt = np.array([[50.0, 50.0]], dtype=np.float64).reshape(-1, 1, 2)
    import cv2
    world_pt = cv2.perspectiveTransform(test_pt, H).reshape(-1, 2)[0]
    assert abs(world_pt[0] - 1.0) < 1e-6 and abs(world_pt[1] - 1.0) < 1e-6, world_pt


def main():
    check("bbox_utils.center() matches hand-computed midpoints", test_center)
    check("bbox_utils.iou() known-overlap cases", test_iou_known_cases)
    check("bbox_utils.bbox_area() including degenerate bbox", test_bbox_area)
    check("bbox_utils.clamp_bbox() clips out-of-bounds + rounds", test_clamp_bbox_clips_out_of_bounds)
    check("homography.bbox_ground_point() is bottom-center, not center", test_bbox_ground_point_is_bottom_center_not_center)
    check("homography.image_to_world() known-scale wiring", test_image_to_world_known_scale)
    check("homography.bbox_to_world() uses the ground point", test_bbox_to_world_uses_ground_point)
    check("homography.CameraCalibration.load() returns None on no active row", test_camera_calibration_load_returns_none_when_no_row)
    check("homography.CameraCalibration.load() happy path via from_row()", test_camera_calibration_load_happy_path)
    check("calibrate_camera._compute_homography() recovers a known scale", test_compute_homography_recovers_known_scale)

    print("\n=== Fase 2 geometry contract verification ===")
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
