"""Camera-to-world coordinate mapping via a single-camera planar homography.

`CameraCalibration.H` maps IMAGE pixel coordinates to WORLD ground-plane
coordinates (in `units`). It is only valid on the ground plane: a homography
is a single 3x3 projective transform, so it cannot recover world position for
points that are not on that plane. `bbox_ground_point()` therefore returns
the bottom-center of a bbox, not its center — for a standing person, the feet
are the only bbox point reliably resting on the ground plane. Using the bbox
center would place a standing person metres away from their true position.

`camera_calibration` rows are produced by `geometry.calibrate_camera`
(manual CLI, run once per camera) and read here via `load()`. That table was
already created (inert) by Fase 0's `postgres_writer.ensure_schema_and_tables()`.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

from camerachatbot.db.postgres_writer import get_conn, SCHEMA


@dataclass
class CameraCalibration:
    camera_id: int
    H: np.ndarray  # 3x3 float64, image -> world (ground plane)
    units: str
    reference_points: list
    reprojection_error: Optional[float]

    @classmethod
    def from_row(cls, row) -> "CameraCalibration":
        """Build a CameraCalibration from a `camera_calibration` table row.

        Expects `row` as a tuple ordered
        `(camera_id, homography, units, reference_points, reprojection_error)`,
        matching the SELECT in `load()`. `homography` is the flat 9-element
        row-major array stored by `calibrate_camera.py`.
        """
        camera_id, homography, units, reference_points, reprojection_error = row
        H = np.asarray(homography, dtype=np.float64).reshape(3, 3)
        return cls(
            camera_id=int(camera_id),
            H=H,
            units=units,
            reference_points=list(reference_points) if reference_points else [],
            reprojection_error=(float(reprojection_error) if reprojection_error is not None else None),
        )

    @classmethod
    def load(cls, camera_id: int) -> Optional["CameraCalibration"]:
        """Load the active calibration for `camera_id`, or None if none exists.

        Graceful degrade by design: callers (Fase 4's zone-classification
        stage) must treat a `None` return as "no calibration for this
        camera" and skip zone/world-coordinate work for it rather than
        raising.
        """
        conn = get_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(f"SET search_path TO {SCHEMA}")
                    cur.execute(
                        """
                        SELECT camera_id, homography, units, reference_points, reprojection_error
                        FROM camera_calibration
                        WHERE camera_id = %s AND is_active
                        LIMIT 1
                        """,
                        (camera_id,),
                    )
                    row = cur.fetchone()
            return cls.from_row(row) if row else None
        finally:
            conn.close()

    def image_to_world(self, pts_xy: np.ndarray) -> np.ndarray:
        """Map (N, 2) image pixel points to (N, 2) world coordinates."""
        pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 1, 2)
        world = cv2.perspectiveTransform(pts, self.H)
        return world.reshape(-1, 2)

    @staticmethod
    def bbox_ground_point(bbox) -> Tuple[float, float]:
        """Bottom-center `(cx, y2)` of an `[x1, y1, x2, y2]` bbox.

        NOT the bbox center: a homography only maps the ground plane, and
        the feet are the only bbox point reliably on it.
        """
        x1, y1, x2, y2 = bbox
        return (0.5 * (x1 + x2), float(y2))

    def bbox_to_world(self, bbox) -> Tuple[float, float]:
        """Ground-point of `bbox` mapped into world coordinates."""
        gx, gy = self.bbox_ground_point(bbox)
        world = self.image_to_world(np.array([[gx, gy]], dtype=np.float64))
        wx, wy = world[0]
        return (float(wx), float(wy))
