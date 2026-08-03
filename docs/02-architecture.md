# Architecture

One paragraph, then the real call chain, stage by stage. If you're extending or debugging the pipeline, this is the map — [`03`](03-pipeline-reference.md) and [`04`](04-security-subsystem-reference.md) go deeper on each module named here.

## Summary

A batch of keyframe images goes in; a set of Postgres rows (detected people/objects, their identities, and their spatial relationships) comes out. Everything in between happens inside one function call, `pipeline_service.run_pipeline_and_persist()`, shared by both HTTP-triggered and local-script runs — the only thing that differs between entry points (covered in [`05`](05-running-the-pipeline.md)) is *how* a folder of images gets there and what happens with the result afterward.

## The data flow, stage by stage

```
run_local.py / run_webhook.py
        │
        ▼
bootstrap.init_runtime()                    — loads every model once (§ Runtime)
        │
        ▼
pipeline_service.run_pipeline_and_persist()
        │
        ├─▶ pipeline_service.build_detail_detectors()   — decides which optional detectors run
        │
        ▼
orchestrator.multi_models()
        │
        ├─ 1. YOLOPersonReID.detect_and_embed()          — detect people/objects, embed each person crop
        ├─ 2. cluster_locally()  OR  assign_track_ids()  — group same person's detections (§ label_source)
        ├─ 3. enforce_unique_label_per_frame()           — drop duplicate labels landing on one frame
        ├─ 4. enroll_and_assign_global_ids()             — resolve against the persistent identity gallery
        ├─ 4b. zones+events stage (security, PR8b)       — zone/intrusion/loitering, dormant in prod (§ below)
        └─ 5. detail detectors loop                       — pose (default), + any flag-enabled extras
        │
        ▼
formatter.reformat_to_video_schema_uniform()   — flat JSON → nested video/frames/objects schema
        │
        ▼
postgres_writer.json_to_postgre()              — batched/threaded insert into Postgres
```

### 1. Runtime bootstrap

`runtime/bootstrap.py::init_runtime()` runs once at process start (or once per import for `run_webhook.py` — see the caveat in [`05`](05-running-the-pipeline.md)). It loads the ONNX detector, ONNX pose classifier, ONNX Re-ID embedder, and optionally the emotion/age ONNX sessions, builds the Supabase client, and assembles one `RUNTIME` dict that everything downstream reads from. This is the single dependency-injection point in the codebase — nothing else loads a model.

### 2. Detection + embedding

`YOLOPersonReID.detect_and_embed()` runs the detector over every frame, filters detections down to a short allowlist of security-relevant classes, and generates a 512-dimensional embedding for every detected person crop. This is the expensive, batched step.

### 3. Grouping detections into people — the `label_source` switch

This is where the codebase's one config-driven fork lives. `SECURITY_RULES["label_source"]` (in `security_config.py`) picks between two ways of deciding "these detections across different frames are the same person":

- **`"cluster"` (the default)**: `cluster_locally()` groups embeddings by similarity within the current batch only. Two appearances of the same person in *different* processing batches aren't linked at this stage — that's what step 4's identity gallery is for.
- **`"tracker"`**: `assign_track_ids()` (built in [`04`](04-security-subsystem-reference.md)) follows detections frame-to-frame through the whole video using a Kalman filter + two-stage matching, surviving brief occlusions. Built and tested, not the default.

Either way, the output is the same shape: a label per detection. Everything after this step doesn't know or care which one produced it.

### 4. Identity resolution

`enforce_unique_label_per_frame()` is a cheap safety net that drops any duplicate label landing twice on the same frame (shouldn't happen, guards against it anyway). `enroll_and_assign_global_ids()` then takes those per-batch labels and resolves each against `GlobalIdentityService`'s persistent FAISS identity gallery — this is what turns "person #3 in this batch" into "person_global_id 47, first seen three weeks ago." It also builds the *neighborhood* records (which people/objects were spatially close to each other).

### 4b. Zones + events (security, PR8b) — wired in, but still dormant in production

As of PR8b, `orchestrator.py` imports `security.zones` and `security.events` directly (previously it only imported `security.tracker`, for step 3's alternate `label_source` path) — confirmed against the code, not assumed. Between step 4 and step 5, it now loads the run's `CameraCalibration`/`Zone` list, tags each person entry with `zone_id`/`world_xy`, builds a per-track timeline, and evaluates intrusion/loitering rules — dumping a `security_events.json` checkpoint next to the other stage checkpoints, and persisting any events to the `event` table via `postgres_writer.insert_events()`.

It's dormant in production today for one specific, tracked reason: the stage takes an optional `camera_id` parameter, and no entry point (`run_local.py`/`run_webhook.py`) resolves a real one before calling `run_pipeline_and_persist()` — camera identity is otherwise only resolved by *name*, lazily, inside `postgres_writer.get_or_create_project_camera_video()` at persistence time. With `camera_id=None`, the stage always takes its documented degrade path (`calib=None`, `zones=[]`), so `evaluate_intrusion`/`evaluate_loitering` run over zero zones and always produce `events=[]`. Threading a real `camera_id` through is the remaining piece of future work here — a missing wire, not a config flag.

The stage is also defensive by design: any failure inside it (bad zone data, `fps=0`, a bug in an evaluator, a checkpoint-write failure) is caught by a top-level exception boundary and degrades to `events=[]` instead of crashing the rest of the run — pose classification and Postgres persistence for unrelated data still succeed even if this stage breaks. See [`04-security-subsystem-reference.md`](04-security-subsystem-reference.md) for the module-level detail, including the `[SECURITY-DEGRADED]` log tag that distinguishes a genuine lookup failure from "this camera just isn't configured yet."

### 5. Detail detectors

A loop over whichever detectors `build_detail_detectors()` decided to build, based on `DETECTOR_FLAGS`. Only `PoseActionClassifier` (sit/stand) is on by default. Each detector reads the previous one's output JSON and adds to it — see [`03`](03-pipeline-reference.md) for the full list and [`08`](08-legacy-and-dormant-code.md) for the ones that are built but flagged off.

### 6. Formatting and persistence

`formatter.reformat_to_video_schema_uniform()` reshapes the flat frame-keyed JSON into the nested schema Postgres expects (deriving each keyframe's timestamp via `video_schema/timing.py::frame_timestamp()`), attaching step 4b's `events` list as a `video.events` node the same way it already does for `neighborhood`. `postgres_writer.json_to_postgre()` then does the actual write: keyframes/objects/classes are inserted synchronously first (later steps need their generated ids), then `insert_events()` runs synchronously right after (see [`04`](04-security-subsystem-reference.md) for why events aren't threaded like metadata/neighborhood), and finally metadata and neighborhood rows are fanned out across a thread pool.

## The three entry points, at a glance

All three call into the same `pipeline_service.run_pipeline_and_persist()` — they differ only in where the input frames come from and what happens to the result. Full detail, including required environment variables, is in [`05-running-the-pipeline.md`](05-running-the-pipeline.md).

| Entry point | Trigger | Feeds into |
|---|---|---|
| `run_local.py` | Manual, `python run_local.py`, no args | A hardcoded local sample folder — smoke-test path |
| `run_webhook.py` | `POST /webhook` (Flask) | Downloads frames from Supabase Storage first |
| `run_live_capture.py` | Manual, `python run_live_capture.py` | Not part of this flow at all — captures from a webcam and uploads to Supabase Storage, which some other process is expected to turn into a `/webhook` call |

## Next

- Module-level detail for everything named above: [`03-pipeline-reference.md`](03-pipeline-reference.md)
- The security subsystem in depth: [`04-security-subsystem-reference.md`](04-security-subsystem-reference.md)
- Database tables this writes to: [`06-database-schema.md`](06-database-schema.md)
