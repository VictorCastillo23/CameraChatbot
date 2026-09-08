"""Pure bbox/geometry helpers shared across the pipeline.

`center()` and `iou()` were originally closures inside
`detectors/person_reid.py::YOLOPersonReID.build_neighborhood()`;
`clamp_bbox()` reproduces the inline clamp-to-image logic used in
`detect_and_embed()`. Extracted here so `build_neighborhood()`, the
Fase 4 tracker (`security/tracker.py`, IoU cost matrix), and any future
geometry consumer share one implementation instead of drifting copies.

`spatial()` stays a local closure in `build_neighborhood()` — it captures
`near_thresh` and is neighborhood-specific, not general-purpose bbox math.

All bboxes are the `[x1, y1, x2, y2]` convention used throughout this
codebase (pixel coordinates, x2/y2 exclusive-ish per the callers' own
clamping — this module does not redefine that contract, just operates on
it).
"""

from typing import Sequence, Tuple


def center(b: Sequence[float]) -> Tuple[float, float]:
    """Return the (cx, cy) center point of an [x1, y1, x2, y2] bbox."""
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def iou(b1: Sequence[float], b2: Sequence[float]) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] bboxes."""
    xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
    xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-12)


def bbox_area(b: Sequence[float]) -> float:
    """Area of an [x1, y1, x2, y2] bbox (0.0 if degenerate/negative)."""
    x1, y1, x2, y2 = b
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    return w * h


def clamp_bbox(b: Sequence[float], W: int, H: int) -> Tuple[int, int, int, int]:
    """Clamp an [x1, y1, x2, y2] bbox to image bounds (W, H), rounding to int.

    Mirrors `detect_and_embed()`'s original inline clamp: each coordinate is
    rounded to the nearest int, then clamped independently to
    `[0, W-1]`/`[0, H-1]`. Callers are responsible for discarding boxes that
    collapse to zero/negative area after clamping (`x2 <= x1 or y2 <= y1`).
    """
    x1, y1, x2, y2 = b
    x1 = max(0, min(W - 1, int(round(float(x1)))))
    x2 = max(0, min(W - 1, int(round(float(x2)))))
    y1 = max(0, min(H - 1, int(round(float(y1)))))
    y2 = max(0, min(H - 1, int(round(float(y2)))))
    return x1, y1, x2, y2
