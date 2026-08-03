# Testing and dev tools

## `tests_manual/`

There is no pytest, no test runner, and no CI config in this repo. Every file under `tests_manual/` is a standalone Python script: plain `assert` statements, a printed PASS/FAIL summary, exit code `0` on success. Run any of them directly:

```bash
python tests_manual/test_identity_thresholds.py
```

| File | Verifies |
|---|---|
| `test_identity_thresholds.py` | The identity-threshold centralization contract — `enroll_and_assign_global_ids()` resolves its thresholds from `security_config.IDENTITY_THRESHOLDS`, no hardcoded stale defaults |
| `test_fase1_attribute_gating.py` | `build_detail_detectors()`'s flag-gating and the `COCO_ALLOWLIST` filter |
| `test_fase2_geometry.py` | `geometry/bbox_utils.py` (center/IoU/area/clamp) and the homography/calibration read path |
| `test_fase3a_onnx_detector_reid.py` | ONNX detector + Re-ID parity against the original torch/ultralytics/torchreid implementations. **Needs real `torch`/`ultralytics`/`torchreid` installed** (from `requirements-export.txt`) to run — the largest test file in the repo |
| `test_fase3a_pose_clustering_parity.py` | ONNX pose classifier + `cosine_clustering()` parity, same real-dependency requirement as above |
| `test_bootstrap_init_runtime.py` | `runtime/bootstrap.py::init_runtime()`'s wiring — monkeypatches the loader/Supabase seams and asserts the `RUNTIME` dict shape |
| `test_tracker.py` | `KalmanBoxTracker`/`ByteTracker` — predict/update round-trips, two-stage association, occlusion recovery, confirmation semantics. Uses synthetic fixtures, no images or models needed |
| `test_zones.py` | `point_in_polygon()` ray-casting (including edge/vertex/concave cases) and `zone_is_armed()`'s schedule logic (including the overnight-wrap-plus-days-filter case) |
| `test_events.py` | `build_tracks_timeline()`, `evaluate_intrusion()`, `evaluate_loitering()` — gap tolerance, dwell-threshold boundaries |

### `tests_manual/fixtures/`

Hand-authored JSON, no images or models needed — the tracker and events tests load these directly:

- `tracks_linear.json`, `tracks_occlusion.json`, `tracks_new_track_needs_min_hits.json`, `tracks_crossing.json`, `tracks_lowconf.json` — synthetic bbox sequences for `test_tracker.py` (a steady track, occlusion recovery, new-track confirmation gating, id-swap avoidance, low-confidence stage-2 keepalive)
- `timeline_intrusion.json`, `timeline_loitering.json` — for `test_events.py`
- `zones_polygon.json` — for `test_zones.py`

## `tools/`

Two dev-only scripts, neither imported by production code.

### `tools/export_to_onnx.py`

Converts the original `.pt` weights to the `.onnx` files the runtime actually loads (see [`05-running-the-pipeline.md`](05-running-the-pipeline.md)):

```bash
python tools/export_to_onnx.py --all
# or selectively:
python tools/export_to_onnx.py --detector
python tools/export_to_onnx.py --posecls --reid
```

Exports `models/yolov10m.pt` → `models/yolov10m.onnx` and `models/trained_yolo11m.pt` → `models/trained_yolo11m.onnx` via Ultralytics' built-in `model.export(format="onnx")`, and the OSNet Re-ID model → `models/osnet_x1_0.onnx` via `torch.onnx.export` (no built-in exporter for that one). Needs `requirements-export.txt` installed — never the runtime environment.

### `tools/verify_onnx_parity.py`

The parity-verification tool used to sign off the ONNX runtime switch: compared the original torch-based detector/Re-ID/pose-classifier outputs against the ONNX replacements on real images, checking IoU/class agreement for the detector, cosine similarity for Re-ID, and class/probability agreement for the pose classifier. User-run, not part of any CI.

**It can no longer be run**: its torch-side comparison target, `runtime/loaders_torch.py`, was deleted once the ONNX switch was confirmed and the torch runtime path was fully removed (see [`08-legacy-and-dormant-code.md`](08-legacy-and-dormant-code.md)). The file is kept in the repo as historical/audit documentation of the parity methodology, not as a runnable tool.
