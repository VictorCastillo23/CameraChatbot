# Code Review Rules

Minimal ruleset distilled from `CLAUDE.md` (the authoritative project-conventions
file already checked into this repo) so the `gga` pre-commit review hook has a
rules file to run against. `CLAUDE.md` remains the source of truth if this
file and `CLAUDE.md` ever disagree.

## Python / general

- No package manager, build step, or test runner exists in this repo.
  `camerachatbot/` is a plain importable package run directly with `python`.
- Never hardcode a CWD-relative path string — resolve everything through
  `camerachatbot/paths.py` (`Path(__file__)`-based), the single source of
  truth for repo-root-relative paths.
- Threshold/config values live in `camerachatbot/security_config.py` only.
  Lower-level functions take them as required keyword-only parameters with
  no default (a missing value should raise `TypeError` immediately, not
  silently fall back to a stale constant).
- Prefer one shared implementation for logic two modules both need (e.g.
  `video_schema/timing.py`'s `frame_timestamp`/`parse_start_at`) over
  reimplementing the same formula twice — two independent implementations
  drift.
- Degrade gracefully instead of raising when optional external state is
  missing (no camera calibration, no zones configured, a DB lookup
  failure) — log a warning and continue with `None`/empty defaults, don't
  abort the whole pipeline run over optional data.
- Code comments, log messages, and JSON field values are in Spanish
  throughout this codebase — this is the established convention, not an
  inconsistency to fix. Module/file/function names are English
  (snake_case).

## Testing

- No pytest. Verification scripts live in `tests_manual/*.py`, run directly
  with `python tests_manual/<name>.py`: plain `assert` statements, a
  `check(name, fn)` harness collecting pass/fail, a PASS/FAIL summary
  printed at the end, and `sys.exit(0/1)`.
- Tests must never touch the repo's real `res/`, `keyFrames/`,
  `gallery.index`/`id_map.json`/`proto_store.npy`, or a live Postgres
  connection unless a test is explicitly and deliberately an integration
  check against real data (and even then, write outputs to a scratch temp
  directory, never the real `res/`/gallery files).

## Architecture

- `camerachatbot/` is a flat package, one subpackage per responsibility
  (`detectors/`, `identity/`, `pipeline/`, `geometry/`, `security/`,
  `video_schema/`, `db/`, etc.) — see `CLAUDE.md`'s architecture table for
  the full module-by-module breakdown.
- `camerachatbot/runtime/bootstrap.py::init_runtime()` is the single
  dependency-injection point: every model/session is loaded once there and
  passed down, not re-loaded per call.
- Config modules (e.g. `security_config.py`) are imported only by
  top-level callers (`orchestrator`, `pipeline_service`, `bootstrap`, CLI
  scripts) — never by `identity/`/`detectors/` internals, which receive
  resolved values as explicit parameters instead.
