# Legacy and dormant code

Two different kinds of "not currently active" in this codebase, easy to conflate: **legacy** is a directory of scripts kept for reference or occasional reuse; **dormant** is fully-integrated production code switched off by a configuration flag. This page covers both.

## `legacy/` — live vs. dead

11 files. Despite older documentation (an earlier version of `CLAUDE.md`) saying only one file is still used, the actual import graph — checked directly, not assumed — shows three:

### Live (still imported by production code)

| File | Imported by | Notes |
|---|---|---|
| `keyframe_extractor_async.py` | `capture/live_capture.py` | Always live — this is `run_live_capture.py`'s actual keyframe extraction logic |
| `face_detector.py` | `pipeline/pipeline_service.py::build_detail_detectors()` | Lazy import, only reached when `DETECTOR_FLAGS["face_attention"]` is `True` (default `False`) |
| `hand_detector.py` | `pipeline/pipeline_service.py::build_detail_detectors()` | Lazy import, only reached when `DETECTOR_FLAGS["hands"]` is `True` (default `False`) |

`face_detector.py` and `hand_detector.py` moved into `legacy/` during the ONNX runtime switch (they depend on `mediapipe`, which was removed from `requirements.txt` when the rest of the ML stack was consolidated onto ONNX Runtime) — they were kept, not deleted, specifically so they could be reactivated without a rewrite. See "Turning a dormant capability back on" below.

### Dead (no imports found anywhere in `camerachatbot/`, `tools/`, `tests_manual/`, or the root entry scripts)

| File | What it was |
|---|---|
| `depth_pose_experiment.py` | Manual MiDaS/YOLO depth experimentation; keeps CWD-relative paths assuming `legacy/` as the working directory — predates the reorg, not touched |
| `keyframe_extractor_mica.py`, `preprocess_mica.py`, `viewer_mica.py` | One of two parallel exploratory capture pipelines — batch processing of existing video files |
| `keyframe_extractor_vivo.py`, `preprocess_vivo.py`, `viewer_vivo.py` | The other exploratory capture pipeline — live camera capture, largely duplicated logic with the "mica" set above |
| `neighborhood_detector.py` | A standalone alternate implementation of neighborhood relations using DeepSort instead of the Re-ID+FAISS approach the production pipeline uses. `deep-sort-realtime` was intentionally dropped from `requirements.txt` specifically because this was its only consumer, and the script already degrades gracefully (`HAS_DEEPSORT=False`) without the package installed |

Treat everything in `legacy/` besides the three live files as exploratory — don't assume it's wired into anything.

## Dormant capabilities — gated by `DETECTOR_FLAGS`

Shipped, tested code that a configuration flag currently keeps out of the default run. All four flags are `False` by default in `security_config.py`.

| Capability | Flag | What it needs to turn on |
|---|---|---|
| Face detection & attention | `DETECTOR_FLAGS["face_attention"]` | Set the flag to `True`; install `mediapipe` manually (see `legacy/face_detector.py`'s own docstring) |
| Hand/gesture detection | `DETECTOR_FLAGS["hands"]` | Same — flag + manual `mediapipe` install |
| Emotion classification | `DETECTOR_FLAGS["emotion"]` | Set the flag to `True`; requires `models/emotion-ferplus-8.onnx` to be present |
| Age estimation | `DETECTOR_FLAGS["age"]` | Set the flag to `True`; requires `models/age_googlenet.onnx` to be present |

Emotion/age share a single detector (`FaceAttributesDetector`) and are built whenever either flag is `True`. If you enable `emotion`/`age` without also enabling `face_attention`, `FaceAttributesDetector` logs a one-time warning — it has no face-localization signal to work from and would otherwise silently produce null results with no indication anything is misconfigured.

## Removed, not dormant

Two capabilities were deleted outright rather than flag-gated — there's nothing to "turn back on" without reverting code:

- **Monocular depth estimation (MiDaS)** — removed as part of consolidating the geometry story onto camera-calibration homographies instead of two separate mechanisms (see [`04-security-subsystem-reference.md`](04-security-subsystem-reference.md)).
- **The torch/ultralytics/torchreid/scikit-learn runtime path** — removed once the ONNX Runtime replacements were verified (see [`07-testing-and-dev-tools.md`](07-testing-and-dev-tools.md)'s note on `tools/verify_onnx_parity.py`). There's no loader module left to fall back to; reverting would mean reverting to a pre-switch commit, not flipping a config value.

The original planning document behind this whole overhaul (`prompt.md`, repo root) is worth reading for the reasoning behind these cuts, and for two capabilities it proposed that were never adopted — see [`01-overview-and-capabilities.md`](01-overview-and-capabilities.md)'s "explicitly out of scope" list.
