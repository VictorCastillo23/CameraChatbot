# Database schema

Every table lives under the Postgres schema `view`, created (if missing) by `db/postgres_writer.py::ensure_schema_and_tables()` on every run — all `CREATE TABLE IF NOT EXISTS`, safe to call repeatedly.

## Core tables — fully wired, read and written on every run

| Table | Purpose |
|---|---|
| `project` | Top-level project/tenant record |
| `camera` | A camera belonging to a project |
| `video` | One ingested video/batch run, tied to a camera |
| `key_frame` | One processed keyframe within a video (timestamp, size, timing stats) |
| `object_class` | Lookup table of distinct detected class names |
| `object` | One detected object instance within a keyframe (class, confidence, bbox, `user_id`) |
| `metadata` | Recursive key/value attribute tree attached to an `object` — self-referential via `metadata_id` as a parent pointer, used for pose/face/hands/emotion/age attributes |
| `neighborhood` | Spatial relation between two objects within a keyframe (intersection, alignment, relation label) |

Notable columns: `object.shape` is a Postgres `BOX` type, not JSON. `metadata.metadata_id` is a self-reference forming a tree, not a foreign key to another table.

## Security tables — mixed wiring status

Created inertly alongside the core tables, but at very different stages of actually being used:

| Table | Schema created | Read | Written | Notes |
|---|---|---|---|---|
| `camera_calibration` | Yes | Yes, as of PR8b — `orchestrator.py`'s zones+events stage calls `CameraCalibration.load(camera_id)`, but only when `camera_id` is set, which no entry point does yet (dormant in production, see [`02`](02-architecture.md)) | Yes — `geometry/calibrate_camera.py`'s manual CLI | Stores a per-camera homography (`DOUBLE PRECISION[]`, 9-element row-major), `reference_points`/`units`/`reprojection_error` |
| `zone` | Yes | Yes, as of PR8b — `orchestrator.py`'s zones+events stage calls `load_zones()`; same `camera_id` dormancy caveat as `camera_calibration` above | **No CLI exists** — unlike `camera_calibration`, there's no tool to insert a zone; it has to be done by hand via SQL | `polygon`/`schedule` are JSONB |
| `authorized_identity` | Yes | No | No | Nothing reads or writes this table anywhere in the codebase yet — reserved for the not-yet-built unauthorized-person detection feature |
| `event` | Yes | No | Yes, as of PR8b — `db/postgres_writer.py::insert_events()`, synchronously right after the `object`/`key_frame` inserts | In practice this table stays empty today: the same `camera_id` dormancy above means the zones+events stage always produces `events=[]` on a real run |

Column notes: `zone.polygon` and `zone.schedule` are JSONB (a list of `[x, y]` pairs and a dict respectively — see [`04-security-subsystem-reference.md`](04-security-subsystem-reference.md) for the exact shape `Zone.from_row()` expects). `event.details` is JSONB, free-form per event type. `event.person_global_id` and `object.user_id` (core table) are both plain, unconstrained integers with no foreign key — `person_global_id` lives in the FAISS gallery's `id_map.json`, not as a Postgres entity, by design.

## Where to look next

- [`04-security-subsystem-reference.md`](04-security-subsystem-reference.md) — the code that reads/writes `camera_calibration` and `zone`, and the code that *would* write `event` once wired in
- [`01-overview-and-capabilities.md`](01-overview-and-capabilities.md) — where "event persistence" and "unauthorized-person detection" sit on the overall capability roadmap
