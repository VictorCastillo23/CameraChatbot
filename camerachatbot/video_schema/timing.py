"""Shared per-frame timestamp derivation (Fase 4b / PR8a).

`formatter.py` already computed per-frame timestamps inline as
``t0 + frame_idx / fps`` for `key_frame.timestamp`. Fase 4b's
`security.events.build_tracks_timeline()` needs the exact same arithmetic
for `TrackObservation.ts`, so it is extracted here rather than reimplemented
a second time — two independent implementations would drift and produce
events whose `started_at` does not line up with any `key_frame.timestamp`
row (the risk the design calls out explicitly).
"""

from datetime import datetime, timedelta


def frame_timestamp(t0: datetime, idx: int, fps: float) -> datetime:
    """Timestamp of frame `idx`, `fps` frames per second after `t0`.

    `idx` is whatever frame-number convention the caller uses consistently:
    `formatter.py` passes `int(frame_id)` (the frame number embedded in the
    source filename), and `events.build_tracks_timeline()` does the same so
    both derivations agree on the same instant for the same frame number.
    """
    return t0 + timedelta(seconds=(float(idx) / float(fps)))
