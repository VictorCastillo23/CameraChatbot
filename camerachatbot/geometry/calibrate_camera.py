"""Manual camera-calibration CLI.

Run once per camera, interactively, to compute the homography that maps
image pixel coordinates to world ground-plane coordinates::

    python -m camerachatbot.geometry.calibrate_camera \\
        --camera-id 1 --image path/to/frame.jpg --units m

Opens an OpenCV window over `--image`. Click >=4 points that lie on the
ground plane; after each click, type the corresponding world coordinate
`wx,wy` (in `--units`) at the terminal prompt. Once >=4 correspondences are
collected, press 'q' to compute `cv2.findHomography(src, dst, cv2.RANSAC)`
and the mean reprojection error over the clicked points, then upsert the
result into `camera_calibration` (deactivating any previously active row for
the same camera). `reference_points` is stored verbatim — image px + world
coords, in click order — so a calibration is auditable and re-runnable
without re-clicking.

This is a manual dev tool: it needs a display and a real ground-plane camera
image, neither of which is available in an automated/CI/sandbox environment.
It has not been exercised interactively — verify it on a real workstation
before relying on it.
"""

import argparse

import cv2
import numpy as np
from psycopg2.extras import Json

from camerachatbot.db.postgres_writer import get_conn, SCHEMA, ensure_schema_and_tables


def _collect_points(image_path: str) -> list:
    """Open an OpenCV window over `image_path`; return clicked correspondences.

    Each entry is `{"image_xy": [x, y], "world_xy": [x, y]}`. Left-click adds
    a point; the matching world coordinate is typed at the terminal prompt
    right after the click. Press 'q' (with >=4 points collected) to finish,
    Esc to abort.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    points = []
    window = "calibrate_camera - click ground points, 'q' to finish, Esc to abort"

    def _on_click(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        cv2.imshow(window, img)
        raw = input(f"World coords for image point ({x}, {y}) as 'wx,wy': ")
        try:
            wx_str, wy_str = raw.split(",")
            world_xy = [float(wx_str.strip()), float(wy_str.strip())]
        except Exception:
            print("[WARN] Could not parse world coordinates ('wx,wy' expected) — discarding this point.")
            return
        points.append({"image_xy": [float(x), float(y)], "world_xy": world_xy})
        print(f"[OK] Collected {len(points)} point(s) so far.")

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _on_click)
    cv2.imshow(window, img)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # Esc
            cv2.destroyAllWindows()
            raise SystemExit("[ABORT] Calibration cancelled by user.")
        if key in (ord("q"), ord("Q")):
            if len(points) < 4:
                print(f"[WARN] Need >= 4 points, have {len(points)}. Keep clicking.")
                continue
            break

    cv2.destroyAllWindows()
    return points


def _compute_homography(points: list):
    """Return `(H, mean_reprojection_error)` from clicked correspondences."""
    src = np.array([p["image_xy"] for p in points], dtype=np.float64)
    dst = np.array([p["world_xy"] for p in points], dtype=np.float64)

    H, _inliers = cv2.findHomography(src, dst, cv2.RANSAC)
    if H is None:
        raise RuntimeError("cv2.findHomography failed to compute a homography from the given points.")

    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    reproj_error = float(np.mean(np.linalg.norm(projected - dst, axis=1)))

    return H, reproj_error


def _upsert_calibration(camera_id: int, H: np.ndarray, units: str,
                         reference_points: list, reprojection_error: float) -> int:
    homography_flat = [float(v) for v in H.reshape(-1)]

    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                ensure_schema_and_tables(cur)
                cur.execute(f"SET search_path TO {SCHEMA}")
                # Deactivate any previously active calibration for this camera —
                # `load()` always resolves the single active row via LIMIT 1.
                cur.execute(
                    "UPDATE camera_calibration SET is_active = FALSE WHERE camera_id = %s AND is_active",
                    (camera_id,),
                )
                cur.execute(
                    """
                    INSERT INTO camera_calibration
                        (camera_id, homography, units, reference_points, reprojection_error, is_active, created_at)
                    VALUES (%s, %s, %s, %s, %s, TRUE, NOW())
                    RETURNING id
                    """,
                    (camera_id, homography_flat, units, Json(reference_points), reprojection_error),
                )
                new_id = cur.fetchone()[0]
        return new_id
    finally:
        conn.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Manual per-camera homography calibration.")
    parser.add_argument("--camera-id", type=int, required=True)
    parser.add_argument("--image", type=str, required=True,
                         help="Path to a ground-plane reference frame for this camera.")
    parser.add_argument("--units", type=str, default="m", help="World coordinate units (e.g. 'm', 'cm').")
    args = parser.parse_args(argv)

    points = _collect_points(args.image)
    H, reproj_error = _compute_homography(points)

    print(f"[OK] Homography computed. Mean reprojection error: {reproj_error:.4f} {args.units}")

    new_id = _upsert_calibration(args.camera_id, H, args.units, points, reproj_error)
    print(f"[OK] camera_calibration row #{new_id} inserted (camera_id={args.camera_id}, active).")


if __name__ == "__main__":
    main()
