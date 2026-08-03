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

### 5. Detail detectors

A loop over whichever detectors `build_detail_detectors()` decided to build, based on `DETECTOR_FLAGS`. Only `PoseActionClassifier` (sit/stand) is on by default. Each detector reads the previous one's output JSON and adds to it — see [`03`](03-pipeline-reference.md) for the full list and [`08`](08-legacy-and-dormant-code.md) for the ones that are built but flagged off.

### 6. Formatting and persistence

`formatter.reformat_to_video_schema_uniform()` reshapes the flat frame-keyed JSON into the nested schema Postgres expects (deriving each keyframe's timestamp via `video_schema/timing.py::frame_timestamp()`). `postgres_writer.json_to_postgre()` then does the actual write: keyframes/objects/classes are inserted synchronously first (later steps need their generated ids), then metadata and neighborhood rows are fanned out across a thread pool.

## Where zones/events would plug in — and why they don't yet

The security subsystem's zone and event logic ([`04`](04-security-subsystem-reference.md)) is not called anywhere in the diagram above. `orchestrator.py` imports `security.tracker` (step 3's alternate path) but never imports `security.zones` or `security.events` — confirmed directly against the code, not assumed. If it were wired in, it would sit as a new stage between step 4 (identity resolution) and step 5 (detail detectors), reading each track's history and camera calibration to classify zone membership and evaluate intrusion/loitering rules. That wiring, plus actually persisting the resulting events to the `event` table, is future work — see [`01`](01-overview-and-capabilities.md)'s capability matrix.

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
