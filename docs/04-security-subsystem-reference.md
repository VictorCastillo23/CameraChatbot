# Security subsystem reference

The modules built specifically for the `security-pipeline-overhaul` program: configuration, continuous tracking, camera calibration, zone/event logic, person authorization, and allowlisted-object detection. This is the part of the codebase evolving fastest, and the part with the most uneven wiring status — each section below states plainly whether the module is actually called by `orchestrator.py`/`pipeline_service.py` today, because that's easy to get wrong by reading the code alone.

## `security_config.py`

The single configuration surface for everything in this file. Convention, stated in its own docstring: imported only by top-level callers (`orchestrator`, `pipeline_service`, `bootstrap`, CLI scripts) — never by `identity/` or `detectors/` internals, which receive resolved values as explicit parameters instead. Every export is frozen (`frozenset`/`tuple`/`MappingProxyType`, including nested dicts) so an accidental in-place mutation (`COCO_ALLOWLIST.add(...)`) raises immediately instead of silently leaking state to every other importer.

| Export | Purpose |
|---|---|
| `COCO_ALLOWLIST` | The 8 class names kept after detection filtering: `person`, `backpack`, `handbag`, `suitcase`, `knife`, `cell phone`, `laptop`, `bottle` |
| `DETECTOR_FLAGS` | Which detail detectors run: `pose=True`, everything else (`face_attention`, `hands`, `emotion`, `age`) `False` |
| `IDENTITY_THRESHOLDS` | Identity-matching thresholds by context: `runtime` (live values, `t_accept=0.80`/`t_reject=0.45`), `enroll` (stricter, for manual enrollment), `cluster` (`eps`/`min_samples` for the default label source), `support` (thresholds for adding extra reference embeddings to a person) |
| `SECURITY_RULES` | `label_source` (`"cluster"` default \| `"tracker"`), `default_loiter_seconds=30`, `intrusion_gap_frames=2`, `unenrolled_debounce_frames=5`, and a nested `tracker` dict of ByteTrack parameters |
| `WEAPONS_DETECTOR` | Config for `detectors/secondary_detector.py::AllowlistedDetector` (Fase 6): `model_path`, `class_names`, `conf`. `model_path=None` (the shipped default) means it stays a genuine no-op — see below |

## `security/tracker.py` — continuous tracking

**Wired in**: yes. `orchestrator.py` imports `assign_track_ids` directly and calls it whenever `SECURITY_RULES["label_source"] == "tracker"` — confirmed by reading `orchestrator.py`'s imports, not assumed. Off by default (`label_source` defaults to `"cluster"`); reachable by changing one config value.

A from-scratch SORT-style Kalman filter (`KalmanBoxTracker`) plus a ByteTrack-style two-stage association tracker (`ByteTracker`), and the `assign_track_ids(reid, cfg)` driver that runs it as a sequential pass over one video's frames — independent of the detector's inference batch size, so a batch boundary never truncates a track.

**Why the two-stage design matters here**: this pipeline's crops come from a security camera feed, where partial occlusion (someone walking behind furniture or another person) is routine. A tracker that discards every low-confidence detection loses the track the instant confidence dips, then re-identifies the same person as a "new" track a few frames later — exactly the kind of churn that would generate spurious enter/leave events in a security context. `ByteTracker.update()` matches high-confidence detections against every track first, then matches remaining low-confidence detections against whatever's still unmatched — recovering an occluded track from a faint detection instead of discarding it or spawning a new id.

`KalmanBoxTracker` tracks a `confirmed` flag: `True` once a track has ever satisfied the reporting threshold (`hits >= min_hits`, or the tracker-startup exception), and it stays `True` for the track's whole life even though the underlying `hits` counter keeps resetting on missed frames. This is what lets a track recovered after an occlusion gap report again immediately with its original id, instead of re-earning `min_hits` from scratch — real ByteTrack/DeepSORT-family "Lost track re-activation" semantics, not the plainer SORT behavior that would otherwise reintroduce the same churn problem the two-stage matching exists to avoid.

## `security/zones.py` — restricted/monitored/safe zone classification

**Wired in**: yes, as of PR8b — `orchestrator.py` imports `load_zones`/`bbox_world_and_zone` directly and calls them in a new stage between identity resolution and the detail-detector loop (see the architecture doc's stage diagram). Still dormant in production today, though: no entry point (`run_local.py`/`run_webhook.py`) resolves a real `camera_id` before calling `run_pipeline_and_persist()`, so the stage always takes the `camera_id=None` degrade path (`calib=None`, `zones=[]`) on every real run — see [`02-architecture.md`](02-architecture.md) for the plumbing gap this leaves.

- `Zone` — a dataclass: `id`, `camera_id`, `name`, `zone_type` (`"restricted"` \| `"monitored"` \| `"safe"`), `polygon` (world coordinates — the same units `CameraCalibration` maps into), `schedule` (optional dict: days/time window/loiter threshold), `is_active`.
- `point_in_polygon(pt, polygon)` — textbook ray-casting, pure Python/NumPy, O(V). Chosen deliberately over OpenCV's `cv2.pointPolygonTest`: world coordinates are floats in metres, and `cv2` wants integer pixel contours — using it here would quantize a metric polygon down to whole units. A point exactly on a polygon edge counts as inside, on purpose: a zone boundary shouldn't silently exclude someone standing on the line.
- `load_zones(camera_id)` — loads active zones for a camera, ordered by `id`. Same graceful-degrade convention as `CameraCalibration.load()` below: returns `[]` if none are configured, rather than raising.
- `zone_is_armed(zone, at)` — `True` if the zone has no schedule (always armed) or `at` falls within the schedule's day-of-week + time window. Correctly handles a schedule that crosses midnight (e.g. `"from": "22:00", "to": "06:00"`): the day-of-week filter is applied to each side of the wrap separately, so a Friday-night-through-Saturday-morning window still reports "armed" at 1am Saturday — a naive same-day check would incorrectly reject that as unarmed, since Saturday itself isn't in a `days: [4]` (Friday) list. This exact bug was found and fixed during development; the fixed logic and a test covering it both exist in the current code.
- `classify_bbox_zone(bbox, calib, zones)` — maps a detection's ground point into world coordinates and returns the first active zone (in list order) whose polygon contains it, or `None`. Deliberately does **not** check `zone_is_armed()` — that's a separate, rule-specific temporal question, left to whichever evaluator needs it (see `evaluate_intrusion` below, the only one that calls `zone_is_armed()`).

## `security/events.py` — intrusion, loitering & unenrolled-person rule evaluation

**Wired in**: yes — `orchestrator.py` calls `build_tracks_timeline()`/`evaluate_intrusion()`/`evaluate_loitering()`/`evaluate_unenrolled()` (the last added in PR9/Fase 5) in the same stage as `zones.py` above, and `db/postgres_writer.py::insert_events()` persists any resulting events to the `event` table synchronously, right after the `object`/`key_frame` inserts (deliberately not through the threaded metadata/neighborhood workers — event volume is low and one event can span a range of frames that doesn't map onto any single worker's partition; `postgres_writer.py` runs on `psycopg` v3 as of PR11, see the note at the end of this file). Same production caveat as `zones.py`: with `camera_id=None`, `evaluate_intrusion`/`evaluate_loitering` always produce nothing (no zones loaded). `evaluate_unenrolled` is camera-independent (see `AuthorizationRegistry` below) but still depends on `build_tracks_timeline` producing a non-empty timeline, which needs the tracker active — see the next bullet.

- `build_tracks_timeline(reid, calib, zones, t0, fps)` — reads a tracked run's detections and groups them per `track_id`, frame-ordered, into a `TracksTimeline`. **Skips any detection without a `track_id`** — which means this function only produces a non-empty timeline when `label_source == "tracker"` was active for the run that produced `reid`'s results (the default `"cluster"` path never sets `track_id`). This dependency isn't stated anywhere else in the code; if zones/events/authorization logic is ever wired into the live pipeline without also flipping `label_source`, it will silently produce empty timelines with no error — this is the reason `evaluate_unenrolled` is dormant on a default run today too, not just `evaluate_intrusion`/`evaluate_loitering`. Degrades gracefully without a camera calibration (`calib=None`): `world_xy`/`zone_id` are `None` on every observation instead of raising, so identity tracking stays usable even before any camera is calibrated.
- `evaluate_intrusion(timeline, zones, cfg)` — one event per continuous run of observations where the zone is `"restricted"` **and** `zone_is_armed()` says so at that observation's timestamp. Tolerates gaps of up to `cfg["intrusion_gap_frames"]` frames within one run (via a shared `_runs()` helper), so one dropped detection doesn't split a single intrusion into several separate events.
- `evaluate_loitering(timeline, zones, cfg)` — one event per zone whose effective `loiter_seconds` (the zone's own schedule, falling back to `cfg["default_loiter_seconds"]`) is set, per continuous run spent in that zone, emitted only once the run's dwell time meets or exceeds the threshold.
- `evaluate_unenrolled(timeline, registry, cfg)` (Fase 5, PR9) — one `event_type="unenrolled_person"` per continuous run where the observation's `person_global_id` is `None` or fails `registry.is_authorized(person_global_id)`. Reported only once a run reaches `>= cfg["unenrolled_debounce_frames"]` **observations** (not wall-clock time, so a low-fps run doesn't arm faster than a high-fps one). Uses the same `_runs()` gap-tolerance primitive as the other two evaluators. See `identity/authorization.py` below for `registry`.
- `evaluate_weapons` is not implemented — the weapons/allowlisted-object detection slot is instead covered by a separate detail detector, `detectors/secondary_detector.py::AllowlistedDetector` (Fase 6, PR10 — see below), which appends `kind="object"` entries to the detection JSON rather than producing `SecurityEvent`s through this module.

## `pipeline/orchestrator.py` — the zones+events+authorization stage (PR8b, PR9)

The wiring that actually calls `zones.py`/`events.py`/`authorization.py` above, sitting between identity resolution and the detail-detector loop — see [`02-architecture.md`](02-architecture.md) for where it fits in the full diagram. Two things worth knowing that aren't obvious from reading `zones.py`/`events.py` alone:

- **Defensive by design, on top of the modules' own graceful degrades**: the whole stage is wrapped in a top-level exception boundary, so a failure anywhere inside it — a malformed zone `schedule` string, `fps=0`, a bug in one of the evaluators, a checkpoint-write failure — degrades to `events=[]` rather than propagating up and crashing the rest of the pipeline run. Pose classification and Postgres persistence for unrelated data still succeed even if this stage breaks. This was gap-review-fixed after an initial version only guarded the calibration/zone *lookups* and let everything downstream of them raise uncaught.
- **`[SECURITY-DEGRADED]` is a distinct, greppable log tag** used specifically when a lookup or the stage itself genuinely fails (e.g. a Postgres outage while loading calibration) — deliberately different from the plain `[WARN]` case, which just means "this camera isn't calibrated/zoned yet." The intent is that an operator or alerting rule scanning logs can tell "not configured" apart from "something broke," which matters because a broken lookup during a real intrusion would otherwise look identical to an unconfigured camera.
- **`AuthorizationRegistry.load()` is called unconditionally** (PR9/Fase 5), regardless of `camera_id` — unlike `CameraCalibration.load(camera_id)`/`load_zones(camera_id)`, since `authorized_identity` has no `camera_id` column. A load failure degrades to `registry=None`, which *skips* `evaluate_unenrolled` entirely rather than falling back to an empty registry — an empty registry would fail-closed for every currently-tracked person and flood false `unenrolled_person` events during what is really a DB outage, not an authorization gap. The `[STAGE 4b]` summary line reports `auth=yes/no` alongside `zonas=`/`tracks=`/`eventos=` (the latter broken down by type: `intrusion=`/`loitering=`/`unenrolled=`).

## `identity/authorization.py` — `AuthorizationRegistry` (Fase 5, PR9)

**Wired in**: yes — loaded unconditionally by `orchestrator.py`'s Fase 4b/5 stage (see above), independent of camera calibration/zone configuration. Still effectively dormant on a default run today for the same root cause as `zones.py`/`events.py`: `evaluate_unenrolled` needs a non-empty `TracksTimeline`, which needs `label_source == "tracker"` (off by default).

Answers "is this person **allowed**", a different question from the FAISS gallery's "have I **seen this body** before" (`identity/global_identity_service.py::GlobalIdentityService`). An immutable snapshot loaded once per pipeline run — batch semantics mean no invalidation is needed, the same convention `CameraCalibration`/`load_zones()` already established. Camera-independent by design: usable before any camera is calibrated.

- `AuthorizationRegistry.load()` — `SELECT ... FROM authorized_identity WHERE is_active`, keyed by `person_global_id`.
- `is_authorized(person_global_id)` — **fail-closed**: `None` or any pid absent from the active-only snapshot (never enrolled, or deauthorized) returns `False`. Deliberate: a fail-open default would make the entire capability a no-op. This is what `evaluate_unenrolled` keys off.
- `get(person_global_id)` — the stored row dict, or `None`.
- Deauthorization is `is_active=False` only (`enroll_person.py --deactivate`, below) — the person stays recognizable in the FAISS gallery and now generates `unenrolled_person` events, which is the intended behavior, not a gap.

## `identity/enroll_person.py` — manual enrollment CLI (Fase 5, PR9)

The only code path that writes to `authorized_identity`. Not part of the live pipeline's import chain — a standalone CLI run manually, once per person to enroll:

```
python -m camerachatbot.identity.enroll_person --name "Ana" --images ./enroll/ana [--role staff]
python -m camerachatbot.identity.enroll_person --deactivate 42
```

`enroll()` loads the runtime, detects the largest person per image, embeds all crops, L2-normalizes the centroid, then calls `gallery.assign_or_create()` with thresholds taken **explicitly** from `IDENTITY_THRESHOLDS["enroll"]` (stricter than the runtime `t_accept`, since this is a manual, higher-confidence operation) — never the live-path default. A gray-zone (ambiguous similarity) match raises `RuntimeError` **before** any `authorized_identity` write, so an uncertain enrollment never silently authorizes the wrong person. If the gallery write (FAISS/`id_map.json`) succeeds but the subsequent `authorized_identity` upsert fails, `enroll()` raises a distinct `RuntimeError` naming the `person_global_id` and stating the person is unauthorized until retried — a documented split-state gap (no gallery rollback primitive exists), not silently swallowed. `main()` wraps both the `enroll()`/`deactivate()` dispatch in `try/except` for `FileNotFoundError`/`RuntimeError`/`psycopg.Error`, printing a clean `[ERROR] ...` message and exiting non-zero instead of a raw traceback.

## `detectors/secondary_detector.py` — `AllowlistedDetector` (Fase 6, PR10)

**Wired in**: yes, conditionally — `pipeline_service.py::build_detail_detectors()` appends it when `WEAPONS_DETECTOR["model_path"] is not None`, a third gating axis distinct from the boolean `DETECTOR_FLAGS` the other detail detectors use (there's no flag for it by design: enabling it without a model path is meaningless). Unconfigured (the shipped default, `model_path=None`) it's a genuine no-op — no `YOLOOnnxDetector` session is ever created, `run_on_json()` returns its input path unchanged, and one warning is logged at construction.

Implements the same detail-detector contract every other detector follows (`frames_folder` set externally by the orchestrator loop, `run_on_json(path) -> path`). When configured, wraps `YOLOOnnxDetector` (Fase 3a) and appends `kind="object"` entries — filtered to `WEAPONS_DETECTOR["class_names"]` — to the detection JSON, writing `tracking_weapons_6.json` alongside the other stage checkpoints.

## `geometry/homography.py` — `CameraCalibration`

**Wired in**: partially. Only the write side (`geometry/calibrate_camera.py`, the manual CLI below) touches the `camera_calibration` table in production code today — the read side, `CameraCalibration.load()`, has no caller outside `zones.py`/`events.py`, which are themselves unwired. Confirmed by grep: nothing in `orchestrator.py`/`pipeline_service.py` imports `CameraCalibration`.

Maps image pixel coordinates to real-world ground-plane coordinates via a single-camera planar homography — a 3×3 transform valid only on the ground plane. `bbox_ground_point()` returns a bbox's bottom-center, not its center: for a standing person the feet are the only bbox point reliably resting on the ground plane, so using the center would place them meters away from their real position.

- `CameraCalibration.load(camera_id)` — loads the active calibration row for a camera, or `None` if none exists. Graceful degrade by design: callers must treat `None` as "skip world-coordinate work for this camera," not an error.
- `image_to_world(pts_xy)` — maps image points to world coordinates.
- `bbox_to_world(bbox)` — the ground-point shortcut most callers actually want.

## `geometry/calibrate_camera.py` — manual calibration CLI

The only code path that writes to the `camera_calibration` table today. Run once per physical camera:

```
python -m camerachatbot.geometry.calibrate_camera --camera-id 1 --image path/to/frame.jpg --units m
```

Opens an OpenCV window over the given image; click ≥4 points on the ground plane, typing each one's real-world coordinate at the terminal prompt. Computes the homography via `cv2.findHomography(..., cv2.RANSAC)` plus a mean reprojection error, then upserts the result (deactivating any prior calibration for the same camera). `reference_points` are stored verbatim so a calibration is auditable and re-runnable without re-clicking.

This is a manual dev tool needing a display and a real ground-plane image — neither available in an automated/CI/sandbox environment. Per its own docstring: **it has not been exercised interactively; verify it on a real workstation before relying on it.**

## `SECURITY_RULES["tracker"]` parameters

Consumed by `assign_track_ids()`/`ByteTracker`, defined in `security_config.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `high_thresh` | 0.5 | Confidence floor for stage-1 matching |
| `low_thresh` | 0.1 | Confidence floor for stage-2 (occlusion-recovery) matching |
| `match_thresh` | 0.8 | Minimum IoU for a detection-to-track match to be accepted |
| `max_age` | 30 | Frames a track survives with no matching detection before being dropped |
| `min_hits` | 3 | Consecutive confirmations a brand-new track needs before being reported |

## Postgres driver note (PR11)

Every module in this file that talks to Postgres directly — `db/postgres_writer.py` (`insert_events`, schema/table creation), `identity/authorization.py` (`AuthorizationRegistry.load()`), and `geometry/calibrate_camera.py` (the `camera_calibration` upsert) — runs on `psycopg` v3 as of PR11, not `psycopg2`. The one behavior-relevant change: `psycopg2.extras.execute_values`'s single-statement multi-row batching (used by `postgres_writer.py`'s bulk inserts) has no v3 equivalent, so it's reimplemented as `execute_values_compat()` in `postgres_writer.py` — same one-statement-per-page guarantee, using parameterized `%s` placeholders instead of `psycopg2`'s `mogrify()`-inlined values (v3's `Cursor` has no `mogrify()`). `identity/enroll_person.py` catches `psycopg.Error`/`psycopg.OperationalError` (not the old `psycopg2` names) around its `authorized_identity` writes. See [`06-database-schema.md`](06-database-schema.md) for the table-by-table detail.
