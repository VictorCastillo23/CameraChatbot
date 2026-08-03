"""Fase 4b (PR8a) — restricted/monitored/safe zone classification.

Zones are polygons in WORLD coordinates (the same units `CameraCalibration`
maps bboxes into, via `camerachatbot.geometry.homography`). `load_zones()`
loads an immutable, `id`-ordered snapshot for a camera once per pipeline
run; `classify_bbox_zone()` then tests a single detection's world ground
point against that snapshot.

`point_in_polygon` is textbook ray-casting (crossing-number algorithm),
pure NumPy/Python, O(V). Chosen deliberately over `cv2.pointPolygonTest`:
world coordinates are floats in metres, and `cv2` wants integer pixel
contours — using it here would quantize a metric polygon down to whole
units. Zone count is small (<20) and points per frame are single digits, so
the asymptotics of a hand-rolled O(V) test are irrelevant.

`classify_bbox_zone()` does NOT check `zone_is_armed()` — it is purely
geometric ("which zone, if any, contains this point"). Whether a zone's
schedule is currently armed is a separate, rule-specific temporal question
(see `security.events.evaluate_intrusion`, which is the only evaluator that
calls `zone_is_armed()`, using each observation's own timestamp).
"""

from dataclasses import dataclass
from datetime import datetime, time as dt_time
from typing import List, Optional, Sequence, Tuple

from camerachatbot.db.postgres_writer import get_conn, SCHEMA
from camerachatbot.geometry.homography import CameraCalibration

Point = Tuple[float, float]


@dataclass
class Zone:
    id: int
    camera_id: int
    name: str
    zone_type: str  # "restricted" | "monitored" | "safe"
    polygon: List[Point]  # world coords, calibration units
    schedule: Optional[dict]  # {"days":[0-6], "from":"22:00", "to":"06:00", "loiter_seconds": 30}
    is_active: bool

    @classmethod
    def from_row(cls, row) -> "Zone":
        """Build a Zone from a `zone` table row.

        Expects `row` ordered `(id, camera_id, name, zone_type, polygon,
        schedule, is_active)`, matching the SELECT in `load_zones()`.
        `polygon`/`schedule` are JSONB, already deserialized by psycopg2 into
        a Python list/dict.
        """
        zone_id, camera_id, name, zone_type, polygon, schedule, is_active = row
        poly = [(float(p[0]), float(p[1])) for p in (polygon or [])]
        return cls(
            id=int(zone_id),
            camera_id=int(camera_id),
            name=name,
            zone_type=zone_type,
            polygon=poly,
            schedule=schedule,
            is_active=bool(is_active),
        )


def _on_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float, eps: float = 1e-9) -> bool:
    """True if `(px, py)` lies on the closed segment `(x1, y1)-(x2, y2)`."""
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if abs(cross) > eps:
        return False
    return (
        min(x1, x2) - eps <= px <= max(x1, x2) + eps
        and min(y1, y2) - eps <= py <= max(y1, y2) + eps
    )


def point_in_polygon(pt: Point, polygon: Sequence[Point]) -> bool:
    """Ray-casting (crossing-number) point-in-polygon test.

    Points exactly on a polygon edge (including a vertex) count as inside —
    a zone boundary must not silently exclude someone standing on the line.
    `polygon` needs >= 3 vertices; fewer always returns False.
    """
    x, y = float(pt[0]), float(pt[1])
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    x1, y1 = float(polygon[-1][0]), float(polygon[-1][1])
    for vx, vy in polygon:
        x2, y2 = float(vx), float(vy)

        if _on_segment(x, y, x1, y1, x2, y2):
            return True

        if (y1 > y) != (y2 > y):
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_intersect:
                inside = not inside

        x1, y1 = x2, y2

    return inside


def load_zones(camera_id: int) -> List[Zone]:
    """Load active zones for `camera_id`, ordered by `id`.

    Graceful degrade by design, same convention as
    `geometry.homography.CameraCalibration.load()`: returns `[]` when no
    zones are configured rather than raising, so Fase 4b's classification
    stage can simply skip zone work for an unconfigured camera.
    """
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(f"SET search_path TO {SCHEMA}")
                cur.execute(
                    """
                    SELECT id, camera_id, name, zone_type, polygon, schedule, is_active
                    FROM zone
                    WHERE camera_id = %s AND is_active
                    ORDER BY id
                    """,
                    (camera_id,),
                )
                rows = cur.fetchall()
        return [Zone.from_row(row) for row in (rows or [])]
    finally:
        conn.close()


def _parse_hhmm(s: str) -> dt_time:
    hh, mm = s.split(":")
    return dt_time(int(hh), int(mm))


def zone_is_armed(zone: Zone, at: datetime) -> bool:
    """True if `zone.schedule` is `None` (always armed), or `at` falls
    within the schedule's day-of-week + time window.

    Correctly handles the overnight-wrap case (`"from": "22:00", "to":
    "06:00"` crosses midnight) — a naive `from <= at <= to` check would
    incorrectly report every such schedule as never armed.
    """
    schedule = zone.schedule
    if not schedule:
        return True

    days = schedule.get("days")
    if days is not None and at.weekday() not in days:
        return False

    from_s = schedule.get("from")
    to_s = schedule.get("to")
    if not from_s or not to_s:
        return True

    t_from = _parse_hhmm(from_s)
    t_to = _parse_hhmm(to_s)
    t_at = at.time()

    if t_from <= t_to:
        return t_from <= t_at <= t_to
    # Overnight wrap: e.g. 22:00 -> 06:00 crosses midnight, so "armed" is
    # everything from `from` to midnight PLUS everything from midnight to
    # `to`, i.e. the complement of the (t_to, t_from) daytime gap.
    return t_at >= t_from or t_at <= t_to


def classify_bbox_zone(bbox, calib: Optional[CameraCalibration], zones: List[Zone]) -> Optional[Zone]:
    """`calib.bbox_to_world(bbox)` -> world point -> test against each
    zone's polygon via `point_in_polygon` -> the FIRST active, containing
    zone in list order, or `None` if there's no calibration or no match.

    Overlapping zones are a configuration mistake, documented rather than
    resolved by priority logic — list order (== `id` order, from
    `load_zones()`) is the only tie-break.
    """
    if calib is None:
        return None

    world_pt = calib.bbox_to_world(bbox)

    for zone in zones:
        if not zone.is_active:
            continue
        if point_in_polygon(world_pt, zone.polygon):
            return zone

    return None
