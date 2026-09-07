# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A computer-vision pipeline that ingests camera keyframes, runs a stack of detection models over them (person detection/Re-ID, pose, face, hands), assigns persistent global identities to people across frames/videos, and persists the result as JSON and then into Postgres. There is no package manager / test runner in this repo — `camerachatbot/` is a plain importable package (no `setup.py`/`pyproject.toml`, no editable install) plus a handful of thin root-level entry scripts, all run directly with `python`.

## Running

There's no build step. Install deps and run an entry point directly from the repo root.

```bash
pip install -r requirements.txt
```

Model weights are **not** in the repo (`models/` is gitignored) — download them from the Google Drive link in `README.md` and place them at:
- `models/yolov10m.pt` (person/object detection)
- `models/trained_yolo11m.pt` (sit/stand pose classifier)
- `models/emotion-ferplus-8.onnx`, `models/age_googlenet.onnx` (ONNX face attribute models)

Required `.env` vars (loaded via `python-dotenv`, at repo root):
- `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_BUCKET` — storage of uploaded frame folders
- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DATABASE`, `POSTGRES_USER`, `POSTGRES_PASSWORD` — analytics DB (`camerachatbot/db/postgres_writer.py`)
- `BATCH_SIZE`, `MAX_WORKERS`, `MAX_RETRIES` — optional, tune `postgres_writer.py` batch/threading behavior (defaults: 1000 / 2 / 3)

Three entry points at the repo root, all run as `python <file>.py` from the repo root (paths resolve relative to the package location via `camerachatbot/paths.py`, not the process's CWD):
- `python run_local.py` — local/offline run: reads images straight from `keyFrames/`, runs the full pipeline once, writes to `res/`, and persists to Postgres. No `POST` involved, no Flask.
- `python run_webhook.py` — Flask app exposing `POST /webhook`. Expects `{"record": {"folder_path", "timestamp", "speed_inference"}}`, downloads that folder from Supabase Storage into `mis_frames/`, runs the same pipeline, and also draws annotated debug images.
- `python run_live_capture.py` — a third, independent entry point: captures frames live from a camera (`cv2.VideoCapture`) via an async producer/consumer, extracts keyframes, and uploads them to Supabase Storage (the counterpart that feeds `run_webhook.py`'s `/webhook`, which some other process is expected to call once a folder finishes uploading). Not part of the `run_local.py`/`run_webhook.py` pipeline itself.

No test suite, linter, or CI config exists in this repo.

## Architecture

`camerachatbot/` is a flat package, one subpackage per responsibility:

| Subpackage/module | Role |
|---|---|
| `camerachatbot/paths.py` | Single source of truth for repo-root-relative paths (models dir, `res/`, `keyFrames/`, `mis_frames/`, the identity-gallery files, the video-schema output path), all resolved via `Path(__file__)` — nothing else in the codebase should hardcode a CWD-relative path string |
| `camerachatbot/runtime/bootstrap.py` (`init_runtime()`) | Loads every model once at process start: YOLO detector + YOLO pose classifier (Ultralytics), OSNet Re-ID model (torchreid), ONNX Runtime sessions for emotion/age, and the Supabase client. Returns everything in one `RUNTIME` dict — the dependency-injection point every entry point calls once |
| `camerachatbot/pipeline/orchestrator.py` (`multi_models(...)`) | The actual detection/tracking/identity pipeline, called once per batch of keyframes (see "Pipeline flow" below) |
| `camerachatbot/pipeline/pipeline_service.py` (`build_detail_detectors()`, `run_pipeline_and_persist()`) | Shared glue between the entry points: builds the 4 detail detectors, calls `orchestrator.multi_models(...)`, reformats the result, and persists it to Postgres. This is what `run_local.py` and `run_webhook.py` both call — keep entry-point-specific behavior (which folder to read, what `video_key`/`size_xy`/`start_at` to use) in the entry points, not here |
| `camerachatbot/detectors/` | `person_reid.py` (`YOLOPersonReID` — detection/embedding/clustering/global-id/neighborhood logic, the largest and most central module), `pose_action_classifier.py`, `face_detector.py`, `hand_detector.py`, `face_attributes_detector.py` |
| `camerachatbot/identity/global_identity_service.py` (`GlobalIdentityService`) | Persistent person identity gallery: FAISS HNSW index (`gallery.index`) + `id_map.json` + `proto_store.npy`. Handles `assign_or_create`, prototype consolidation, and gray-zone (ambiguous similarity) handling |
| `camerachatbot/video_schema/formatter.py` (`reformat_to_video_schema_uniform()`) | Flat detection JSON → nested `video.key_frames[].objects[].metadata` schema consumed by Postgres |
| `camerachatbot/db/postgres_writer.py` (`json_to_postgre()`) | Schema creation + batched/threaded Postgres ingestion |
| `camerachatbot/storage/bucket_uploader.py` | Uploads captured frames to Supabase Storage + logs to `uploaded_folders` table (used by the live-capture path) |
| `camerachatbot/storage/bucket_downloader.py` (`download_folder()`) | Downloads a folder of frames from Supabase Storage (used by `run_webhook.py`) |
| `camerachatbot/preprocessing/image_preprocessing.py` (`preprocess_image()`) | Resizes images to keep aspect ratio and dimensions as multiples of 32 (model input requirement) |
| `camerachatbot/debugging/annotate.py` (`draw_json_over_folder()`) | Debug-only: draws bboxes/attributes over frame images for visual QA |
| `camerachatbot/capture/live_capture.py` | Async producer/consumer that captures from a live camera and uploads keyframes — the logic behind `run_live_capture.py` |
| `camerachatbot/geometry/` | `bbox_utils.py` (pure bbox math: `center()`, `iou()`, `bbox_area()`, `clamp_bbox()`, shared by `person_reid.py` and future tracking code), `homography.py` (`CameraCalibration` — loads a per-camera image-to-world homography from the `camera_calibration` table and maps bbox ground points into world coordinates), `calibrate_camera.py` (manual CLI to compute and store a camera's homography by clicking image/world point correspondences) |

### Pipeline flow (`run_local.py` and `run_webhook.py` both follow this shape via `pipeline_service.run_pipeline_and_persist()`)

1. `bootstrap.init_runtime()` loads every model once (see table above).
2. `orchestrator.multi_models(...)`:
   - `YOLOPersonReID.detect_and_embed()` — batched YOLO inference over every frame, ReID embedding per person crop. Writes an intermediate JSON keyed by frame id.
   - `YOLOPersonReID.cluster_locally()` — DBSCAN over ReID embeddings (cosine) to group same-looking people **within this batch** into local `track_id`s.
   - `YOLOPersonReID.enforce_unique_label_per_frame()` — drops duplicate cluster labels landing on the same frame.
   - `YOLOPersonReID.enroll_and_assign_global_ids()` — resolves each local cluster against `GlobalIdentityService` (persistent FAISS gallery) to get a `person_global_id` that's stable **across batches/videos**; also builds the `neighborhood` (spatial relations between people/objects) via `build_neighborhood()`.
   - Then each **detail detector** (built by `pipeline_service.build_detail_detectors()`, individually gated by `security_config.DETECTOR_FLAGS`) runs in sequence over the same JSON, each reading the previous one's output and adding to `attributes`: `PoseActionClassifier` (sit/stand, flag `pose`), `FaceDetector` (MediaPipe FaceMesh → attention/mouth, flag `face_attention`), `HandDetector` (MediaPipe Hands → gestures, flag `hands`), `FaceAttributesDetector` (ONNX emotion + age on face crops, flags `emotion`/`age`, built when either is `True`). Only detectors whose flag is `True` are built and run — the shipped default (`pose=True`, everything else `False`) runs just `PoseActionClassifier`.
3. `formatter.reformat_to_video_schema_uniform()` reshapes the flat frame→entries JSON into the final nested schema (timestamps derived from `fps` + `start_at`, metadata trees for face/hands/pose).
4. `postgres_writer.json_to_postgre()` loads that schema into Postgres under the `view` schema: creates tables if missing, batch-inserts `key_frame`/`object`/`object_class` synchronously (needed because later steps depend on generated ids), then fans out `metadata` + `neighborhood` inserts across `MAX_WORKERS` threads (one Postgres connection per thread, one keyframe range per thread) with deadlock retry/backoff.
5. Each detector writes its own JSON checkpoint (`tracking_face_3.json`, `tracking_pose_2.json`, etc. — the numeric suffix reflects pipeline stage order, not versioning), all under the batch's `res/<frames_folder_basename>/yolo_reid/` directory.

### `legacy/` — experimental capture scripts, not part of the production pipeline

A namespace package (no `__init__.py`, same as before) of manually-run, exploratory scripts — kept runnable but never imported by anything under `camerachatbot/` except `keyframe_extractor_async.py` (see below). Contains two parallel, largely-duplicated keyframe-capture experiments (foreground-change-based, via background subtraction + peak finding):
- `viewer_vivo.py` → `keyframe_extractor_vivo.py` → `preprocess_vivo.py` — live camera capture (async producer/consumer, `cv2.VideoCapture`)
- `viewer_mica.py` → `keyframe_extractor_mica.py` → `preprocess_mica.py` — batch processing of existing video files
- `neighborhood_detector.py` — a standalone, unused alternate implementation of neighborhood relations using DeepSort instead of the ReID+FAISS approach
- `depth_pose_experiment.py` — manual MiDaS/YOLO experimentation; keeps pre-existing CWD-relative paths that assume it's run with `legacy/` as the working directory (do not "fix" these, they predate the reorg and aren't part of production)
- `keyframe_extractor_async.py` — the one file in `legacy/` actually used in production, imported by `camerachatbot/capture/live_capture.py`

Treat this directory as exploratory — don't assume anything in it besides `keyframe_extractor_async.py` is wired into the production pipeline.

### Identity persistence gotcha

`gallery.index`, `id_map.json`, and `proto_store.npy` are stateful, gitignored files in the repo root (paths come from `camerachatbot/paths.py`) that persist person identities **across runs**. Deleting one without the others desyncs the FAISS index from the id map — `GlobalIdentityService.__init__` already detects and repairs count mismatches on load, but be aware that wiping these resets all known identities.

### Language note

Code comments, log messages, and JSON field values (e.g. `"sentado"`, `"de pie"`) are in Spanish throughout this codebase — this is the established convention, not an inconsistency to fix. Module/file names are English (snake_case), following this same established split.
