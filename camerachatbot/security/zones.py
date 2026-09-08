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

    The `days` filter is applied per side of the wrap, not against `at`
    alone: for a wrapping window, a day in `days` arms BOTH the tail end of
    that day (`from` to midnight) AND the head end of the FOLLOWING day
    (midnight to `to`). Applying the `days` filter to `at` before branching
    on the wrap (the previous, buggy behavior) incorrectly rejected the
    post-midnight continuation of an overnight window, e.g.
    `{"days": [4], "from": "22:00", "to": "06:00"}` (armed Friday night
    through Saturday morning) would report Saturday 01:00 as unarmed,
    because Saturday's weekday isn't in `days` — even though it's the
    intended continuation of Friday night's window.
    """
    schedule = zone.schedule
    if not schedule:
        return True

    days = schedule.get("days")

    from_s = schedule.get("from")
    to_s = schedule.get("to")
    if not from_s or not to_s:
        # No time window configured -- only the day-of-week filter (if any)
        # applies.
        if days is not None and at.weekday() not in days:
            return False
        return True

    t_from = _parse_hhmm(from_s)
    t_to = _parse_hhmm(to_s)
    t_at = at.time()

    if t_from <= t_to:
        if days is not None and at.weekday() not in days:
            return False
        return t_from <= t_at <= t_to

    # Overnight wrap: e.g. 22:00 -> 06:00 crosses midnight. Without a `days`
    # filter, "armed" is everything from `from` to midnight PLUS everything
    # from midnight to `to` (the complement of the (t_to, t_from) daytime
    # gap). With a `days` filter, each side of the wrap is checked against
    # the day it actually belongs to: the tail end (`t_at >= t_from`)
    # belongs to `at`'s own weekday; the head end (`t_at <= t_to`) belongs
    # to the PRECEDING day's overnight window, i.e. `at.weekday() - 1`.
    if days is None:
        return t_at >= t_from or t_at <= t_to
    if t_at >= t_from:
        return at.weekday() in days
    if t_at <= t_to:
        return (at.weekday() - 1) % 7 in days
    return False


def bbox_world_and_zone(
    bbox, calib: Optional[CameraCalibration], zones: List[Zone]
) -> Tuple[Optional[Point], Optional[Zone]]:
    """Single computation of a bbox's world ground point AND its classified
    zone (PR8b), returning `(world_xy, zone)`.

    Both `orchestrator._write_zone_and_world_xy()` and `security.events.
    build_tracks_timeline()` need exactly this pair for the same bbox at the
    same instant. Before this helper existed, each computed it independently
    via its own `calib.bbox_to_world(bbox)` + `classify_bbox_zone(bbox, ...)`
    calls -- two (really three, counting `classify_bbox_zone`'s own internal
    `bbox_to_world` call) separate computations of the same formula that
    could silently drift out of sync (a future change to either call site --
    a different zone-priority tiebreak, a caching layer, a coordinate-system
    tweak -- could desync a person entry's `zone_id`/`world_xy` from its
    `TrackObservation`'s `zone_id`/`world_xy`, with nothing to catch it).
    One implementation instead of two, same anti-drift principle as
    `video_schema.timing.frame_timestamp`/`parse_start_at`.

    `world_xy` is `None` when `calib is None`. `zone` is the FIRST active
    zone (in list order == `id` order, from `load_zones()`) whose polygon
    contains `world_xy`, or `None` if there's no calibration or no match.
    Overlapping zones are a configuration mistake, documented rather than
    resolved by priority logic.
    """
    if calib is None:
        return None, None

    world_xy = calib.bbox_to_world(bbox)

    for zone in zones:
        if not zone.is_active:
            continue
        if point_in_polygon(world_xy, zone.polygon):
            return world_xy, zone

    return world_xy, None


def classify_bbox_zone(bbox, calib: Optional[CameraCalibration], zones: List[Zone]) -> Optional[Zone]:
    """`calib.bbox_to_world(bbox)` -> world point -> test against each
    zone's polygon via `point_in_polygon` -> the FIRST active, containing
    zone in list order, or `None` if there's no calibration or no match.

    Overlapping zones are a configuration mistake, documented rather than
    resolved by priority logic — list order (== `id` order, from
    `load_zones()`) is the only tie-break.

    Thin wrapper over `bbox_world_and_zone()` (PR8b) for callers that only
    need the zone, not the world point -- kept as a separate public function
    since it predates PR8b (Fase 4b/PR8a) and `tests_manual/test_zones.py`
    already tests it directly by this name/signature.
    """
    _, zone = bbox_world_and_zone(bbox, calib, zones)
    return zone
