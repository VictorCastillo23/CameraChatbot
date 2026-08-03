# Pipeline reference

Module-by-module reference for the code that runs on every request today — detection, identity, formatting, storage, and support utilities. For the security-specific modules (tracking, zones, events, calibration), see [`04-security-subsystem-reference.md`](04-security-subsystem-reference.md). For the end-to-end story these modules fit into, see [`02-architecture.md`](02-architecture.md).

## `paths.py`

Single source of truth for every repo-root-relative path (models dir, `res/`, `keyFrames/`, `mis_frames/`, the identity-gallery files, the video-schema output path). Every path is resolved via `Path(__file__)`, not the process's current working directory — nothing else in the codebase should hardcode a CWD-relative path string.

## `runtime/bootstrap.py` — `init_runtime()`

The single dependency-injection point: called once, loads every model and client, returns one `RUNTIME` dict everything downstream reads from.

- `init_runtime()` — builds the Supabase client, the Flask `app`, the ONNX detector + pose classifier + Re-ID embedder (via `runtime/loaders_onnx.py`, see below), and the emotion/age ONNX sessions (only if their flags are on). Returns `RUNTIME` with keys: `app`, `supabase`, `BUCKET_NAME`, `yolo_det`, `yolo_posecls`, `reid_model`, `emotion`, `age`.
- `onnx_providers()` — CUDA execution provider first if available, else CPU-only. This is the *only* device-selection logic in the codebase; there is no separate torch device concept anymore.
- `build_supabase()` — tolerant: if `SUPABASE_URL`/`SUPABASE_KEY` aren't set, logs a warning and returns `(None, None)` instead of raising.
- `load_face_attr_sessions()` — skips loading the emotion/age ONNX sessions entirely when both `DETECTOR_FLAGS["emotion"]` and `["age"]` are `False` (the default), saving startup time and memory for a capability that isn't in use.

## `runtime/loaders_onnx.py`

The sole model-loading module in the codebase (see [`08-legacy-and-dormant-code.md`](08-legacy-and-dormant-code.md) for what used to exist alongside it). Exposes `load_detector()`, `load_posecls()`, `load_reid()` — each resolves its `.onnx` file under `models/` (see [`05`](05-running-the-pipeline.md) for exactly which files), raises `FileNotFoundError` with a hint to run the export tool if the file is missing, and constructs the matching wrapper class from `detectors/`.

## `pipeline/orchestrator.py` — `multi_models(...)`

The actual detection/identity pipeline body, called once per batch of keyframes. Resolves identity thresholds from `security_config.IDENTITY_THRESHOLDS` (any left as `None` by the caller), runs detection+embedding, branches on `SECURITY_RULES["label_source"]` to decide whether `cluster_locally()` or the tracker's `assign_track_ids()` produces per-detection labels, resolves those labels into persistent identities, runs the zones+events security stage (PR8b — takes `camera_id`/`start_at`/`fps`, dormant in production until a real `camera_id` is threaded in, see [`04`](04-security-subsystem-reference.md)), and finally runs the detail-detector loop. Returns a 4-tuple, `(json_output, events, final_pre_process, final_post_process)` — `events` is new as of PR8b. See [`02-architecture.md`](02-architecture.md) for the full stage-by-stage walkthrough — this function *is* that diagram.

## `pipeline/pipeline_service.py`

Shared glue between the entry points.

- `build_detail_detectors(runtime, frames_folder)` — builds the list of detail detectors whose `DETECTOR_FLAGS` entry is `True`. `FaceDetector`/`HandDetector` are imported lazily, inside the `if` branch, only when their flag is on — not at module load time — so importing this file doesn't require `mediapipe` to be installed at all unless those flags are flipped.
- `run_pipeline_and_persist(...)` — the function both `run_local.py` and `run_webhook.py` call: builds detail detectors, runs `orchestrator.multi_models()`, reformats the result, persists it to Postgres, and optionally draws debug-annotated images. Entry-point-specific behavior (which folder to read, what `video_key`/`size_xy`/`start_at` to use) lives in the entry points, not here.

## `detectors/person_reid.py` — `YOLOPersonReID`

The largest and most central module in the codebase. One instance is constructed per pipeline run.

- `detect_and_embed(allowlist=...)` — batched detector inference over every frame in the folder; for every detected person, extracts a Re-ID embedding via the wired `reid_model`. Detections whose class isn't in `allowlist` (`security_config.COCO_ALLOWLIST`) are discarded, not just hidden. Writes an intermediate JSON keyed by frame id, and records `self.frame_order` (the frame processing order, reused by the tracker — see [`04`](04-security-subsystem-reference.md)).
- `cluster_locally(eps, min_samples)` — the default label source: groups this batch's embeddings by cosine similarity via `identity/cosine_clustering.py::cluster_cosine()` (a from-scratch, dependency-free replacement for scikit-learn's `DBSCAN`, verified to match it exactly on synthetic data).
- `enforce_unique_label_per_frame(labels)` — drops duplicate labels landing on the same frame.
- `enroll_and_assign_global_ids(gallery, labels, *, t_accept, t_reject, min_sim_add, max_protos_per_person)` — resolves each local label against the persistent `GlobalIdentityService` gallery to get a `person_global_id` stable across batches/videos. All four keyword args are required, no defaults — a caller that forgets one gets an immediate `TypeError` rather than silently inheriting a stale threshold.
- `build_neighborhood(...)` — computes spatial relations (proximity, alignment) between detected people/objects in the same frame.

## `identity/global_identity_service.py` — `GlobalIdentityService`

The persistent person-identity gallery: a FAISS HNSW index (`gallery.index`) plus `id_map.json` and `proto_store.npy` on disk. See the "identity persistence gotcha" note in [`05-running-the-pipeline.md`](05-running-the-pipeline.md) before deleting any of these files individually.

- `assign_or_create(emb, modality="body", *, t_accept, t_reject, ...)` — the core matching call: given an embedding, either matches it to an existing person or creates a new one.
- `search(emb, k=10)` — nearest-neighbor lookup against the gallery.
- `add_prototypes(person_global_id, prototypes)`, `maybe_add_support_prototype(...)` — grow a person's set of reference embeddings over time.
- `consolidate_person(pid, keep_k=4, ...)` — prunes a person's prototypes down to a representative subset.

## `identity/cosine_clustering.py` — `cluster_cosine()`

A pure-numpy, dependency-free drop-in for `sklearn.cluster.DBSCAN(eps, min_samples, metric="cosine").fit_predict(E)`. Written specifically so the `scikit-learn` runtime dependency could be dropped entirely once the ONNX runtime switch (see [`07`](07-testing-and-dev-tools.md) for how that was verified) was confirmed safe. Same return contract: an `(N,)` int array, `-1` for noise points. Cluster label *numbering* and which cluster an ambiguous border point lands in aren't guaranteed to match scikit-learn's — neither does scikit-learn itself guarantee that, by its own documentation.

## `detectors/onnx_yolo.py` — `YOLOOnnxDetector`

The ONNX Runtime replacement for the original Ultralytics YOLO detector. Duck-types the Ultralytics result shape (`Box`/`DetResult` with `.xyxy`/`.conf`/`.cls`/`.names`) so callers needed zero changes when this replaced the original. Handles two possible ONNX output layouts: the production detector's NMS-free `(1, 300, 6)` layout, and a v8-style `(1, 84, 8400)`+NMS layout (with per-class non-max suppression) for any other export shape. Letterbox preprocessing (`_letterbox`) replicates Ultralytics' own resize/pad behavior, including matching its choice between square-padding and shape-preserving padding depending on whether a batch's images share one original size.

## `detectors/onnx_reid.py` — `OSNetOnnxEmbedder`

The ONNX Runtime replacement for the original torchreid OSNet embedder. `_preprocess()` replicates torchreid's exact resize/normalize pipeline. `embed()`/`embed_one()` return L2-normalized embeddings; a crop that fails to preprocess produces an all-zero (not unit-norm) row rather than raising, with `is_valid_embedding()` provided so callers doing cosine similarity can filter those out explicitly instead of silently comparing against a zero vector.

## `detectors/onnx_pose_classifier.py` — `PoseClsOnnxClassifier`

The ONNX Runtime replacement for the original Ultralytics YOLO-CLS sit/stand classifier. Not a thin wrapper — Ultralytics' classification pipeline handles batching/precision/preprocessing internally in ways ONNX Runtime doesn't provide for free, so this reimplements the resize-then-center-crop preprocessing (matched to Pillow's own resize filter, since that's what Ultralytics' real preprocessing path uses) and the batching loop directly.

## `detectors/pose_action_classifier.py` — `PoseActionClassifier`

The detail-detector wrapper that turns `PoseClsOnnxClassifier`'s raw predictions into the pipeline's `"sentado"`/`"de pie"` (sit/stand) labels, gated by `DETECTOR_FLAGS["pose"]` (on by default).

## `detectors/face_attributes_detector.py` — `FaceAttributesDetector`

Runs the emotion and age ONNX models over face crops. Built when either `DETECTOR_FLAGS["emotion"]` or `["age"]` is `True` — off by default (see [`08`](08-legacy-and-dormant-code.md)). Logs a one-time-per-process warning if enabled while `face_attention` is off, since it then has no face-localization signal to work from and would otherwise silently produce `label=None, confidence=0.0` with no indication anything is misconfigured.

## `video_schema/formatter.py` — `reformat_to_video_schema_uniform()`

Reshapes the flat frame-keyed detection JSON into the nested `video.key_frames[].objects[].metadata` schema Postgres expects. Uses `video_schema/timing.py::frame_timestamp()` for per-frame timestamps rather than computing them inline.

## `video_schema/timing.py` — `frame_timestamp(t0, idx, fps)`

`t0 + idx / fps`, extracted into its own module specifically so `formatter.py` and the security-events logic ([`04`](04-security-subsystem-reference.md)) compute timestamps identically — two independent implementations would drift and produce events whose timestamps don't line up with any `key_frame` row.

## `storage/bucket_downloader.py` / `storage/bucket_uploader.py`

Supabase Storage I/O. `download_folder()` pulls a frame folder down (used by `run_webhook.py`); `upload_cv2_images_to_folder()` pushes captured frames up (used by the live-capture path, `capture/live_capture.py`).

## `preprocessing/image_preprocessing.py` — `preprocess_image()`

Resizes an image to keep its aspect ratio while making both dimensions multiples of 32 — a model input requirement.

## `debugging/annotate.py` — `draw_json_over_folder()`

Debug-only: draws bounding boxes and attribute labels over frame images for visual QA. Invoked when `run_pipeline_and_persist(..., draw_debug=True)` — `run_webhook.py` always passes this, `run_local.py` doesn't by default.

## `capture/live_capture.py`

Async producer/consumer that captures frames from a live camera (`cv2.VideoCapture`), extracts keyframes via `legacy/keyframe_extractor_async.py`, and uploads them to Supabase Storage. This is the logic behind `run_live_capture.py` — see [`05-running-the-pipeline.md`](05-running-the-pipeline.md) for how it relates (loosely — no automatic hand-off exists) to `run_webhook.py`.
