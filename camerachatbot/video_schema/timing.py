"""Shared per-frame timestamp derivation (Fase 4b / PR8a) and start-time
parsing (Fase 4b / PR8b).

`formatter.py` already computed per-frame timestamps inline as
``t0 + frame_idx / fps`` for `key_frame.timestamp`. Fase 4b's
`security.events.build_tracks_timeline()` needs the exact same arithmetic
for `TrackObservation.ts`, so it is extracted here rather than reimplemented
a second time — two independent implementations would drift and produce
events whose `started_at` does not line up with any `key_frame.timestamp`
row (the risk the design calls out explicitly).

PR8b extracts `parse_start_at()` for the same reason: `orchestrator.
multi_models()`'s new zones+events stage (Fase 4b) needs to parse the run's
`start_at` string into the same `t0` `formatter.py` uses, so both agree on
which instant frame 0 actually is.
"""

import re
from datetime import datetime, timedelta, timezone


def frame_timestamp(t0: datetime, idx: int, fps: float) -> datetime:
    """Timestamp of frame `idx`, `fps` frames per second after `t0`.

    `idx` is whatever frame-number convention the caller uses consistently:
    `formatter.py` passes `int(frame_id)` (the frame number embedded in the
    source filename), and `events.build_tracks_timeline()` does the same so
    both derivations agree on the same instant for the same frame number.
    """
    return t0 + timedelta(seconds=(float(idx) / float(fps)))


def parse_start_at(ts: str) -> datetime:
    """Parses a run's `start_at` string into a timezone-aware (UTC) datetime.

    Accepts a trailing `Z` (UTC shorthand) or an explicit offset; a
    fractional-seconds component shorter than microsecond precision (e.g.
    `.12+00:00`) is right-padded to 6 digits before `datetime.fromisoformat`,
    which only accepts 3 or 6. A naive datetime (no offset at all) is
    assumed to already be UTC.

    Extracted from `formatter.reformat_to_video_schema_uniform()`'s former
    inline `_parse_start` closure (PR8b) so `orchestrator.multi_models()`'s
    Fase 4b zones+events stage parses `start_at` identically — one
    implementation instead of two that could drift on which `t0` a track's
    event timestamps are computed from.
    """
    if ts.endswith("Z"):
        dt = datetime.fromisoformat(ts[:-1]).replace(tzinfo=timezone.utc)
    else:
        ts_fixed = re.sub(r'(\.\d{1,5})(\+|\-)', lambda m: f"{m.group(1).ljust(7, '0')}{m.group(2)}", ts)
        dt = datetime.fromisoformat(ts_fixed)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
    return dt
