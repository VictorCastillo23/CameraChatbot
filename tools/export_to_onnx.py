"""Dev-only ONNX export tool — offline, run once per model update.

Exports the three torch models the pipeline uses today into the ONNX files
`camerachatbot/detectors/onnx_yolo.py::YOLOOnnxDetector` and
`camerachatbot/detectors/onnx_reid.py::OSNetOnnxEmbedder` consume:

    models/yolov10m.pt        -> models/yolov10m.onnx        (ultralytics export)
    models/trained_yolo11m.pt -> models/trained_yolo11m.onnx (ultralytics export)
    osnet_x1_0 (torchreid)    -> models/osnet_x1_0.onnx      (torch.onnx.export)

NEVER imported by production code — nothing under `camerachatbot/` imports
this file, and it is not part of the `run_local.py`/`run_webhook.py`
pipeline. Requires the torch/ultralytics/torchreid/onnx stack pinned in
`requirements-export.txt` at the repo root, which is intentionally kept out
of the runtime `requirements.txt` (Fase 3c drops that stack from the
runtime environment entirely, once the ONNX switch in `bootstrap.py` has
been flipped and verified via `tools/verify_onnx_parity.py`).

Usage:
    python tools/export_to_onnx.py --all
    python tools/export_to_onnx.py --detector
    python tools/export_to_onnx.py --posecls --reid
"""

import argparse
from pathlib import Path

import onnx
import torch
import torchreid
from ultralytics import YOLO

from camerachatbot import paths

REID_ONNX_PATH = paths.MODELS_DIR / "osnet_x1_0.onnx"


def _ensure_names_metadata(onnx_path: Path, names: dict) -> None:
    """Guarantee `names` is present in the exported model's `metadata_props`.

    `onnx_yolo.py::YOLOOnnxDetector._read_names_metadata()` reads
    `sess.get_modelmeta().custom_metadata_map["names"]` and falls back to
    the generic 80-class `COCO_NAMES` constant when it's absent — a wrong or
    missing `names` map silently breaks `COCO_ALLOWLIST` filtering, and
    would be actively wrong for `trained_yolo11m.pt` (sit/stand classes,
    NOT COCO).

    In practice this is a no-op: `ultralytics.engine.exporter.Exporter`
    already writes `model.names` into `self.metadata["names"]` and then
    serializes EVERY `self.metadata` key into the ONNX file's
    `metadata_props` via `meta = model_onnx.metadata_props.add(); meta.key,
    meta.value = k, str(v)` (confirmed by reading the installed
    `ultralytics==8.4.115` package source,
    `ultralytics/engine/exporter.py::Exporter.export_onnx`). This function
    exists purely as a defensive safety net against a future ultralytics
    version changing that behavior — it only touches the file if `"names"`
    is missing.
    """
    model = onnx.load(str(onnx_path))
    existing_keys = {p.key for p in model.metadata_props}
    if "names" in existing_keys:
        return
    prop = model.metadata_props.add()
    prop.key, prop.value = "names", str(names)
    onnx.save(model, str(onnx_path))


def export_detector(imgsz: int = 640, dynamic: bool = True) -> Path:
    """Export `models/yolov10m.pt`.

    YOLOv10 has an NMS-free head, so the exported ONNX output is
    `(N, 300, 6)` = `[x1, y1, x2, y2, conf, cls]` — the layout
    `onnx_yolo.py::YOLOOnnxDetector._postprocess()`'s `nms_free` branch
    expects.
    """
    pt_path = paths.MODELS_DIR / "yolov10m.pt"
    model = YOLO(str(pt_path))
    exported = model.export(format="onnx", imgsz=imgsz, dynamic=dynamic)
    onnx_path = Path(exported)
    _ensure_names_metadata(onnx_path, model.names)
    print(f"[export] detector -> {onnx_path}")
    return onnx_path


def export_posecls(imgsz: int = 224, dynamic: bool = True) -> Path:
    """Export `models/trained_yolo11m.pt` (YOLO-CLS sit/stand classifier).

    Consumed by `camerachatbot/detectors/onnx_pose_classifier.py`
    (Fase 3a-part-2 / PR4b — not part of this PR), exported here anyway
    since this tool covers all three models in one pass.
    """
    pt_path = paths.MODELS_DIR / "trained_yolo11m.pt"
    model = YOLO(str(pt_path))
    exported = model.export(format="onnx", imgsz=imgsz, dynamic=dynamic)
    onnx_path = Path(exported)
    _ensure_names_metadata(onnx_path, model.names)
    print(f"[export] pose classifier -> {onnx_path}")
    return onnx_path


def export_reid(height: int = 256, width: int = 128, dynamic: bool = True) -> Path:
    """Export the OSNet Re-ID backbone via `torch.onnx.export`.

    Ultralytics has no involvement with this model (it comes from
    `torchreid.models.build_model(name="osnet_x1_0", ...)`, matching
    `camerachatbot/runtime/bootstrap.py::load_reid()`), so there is no
    built-in exporter here — this is the manual path.
    """
    reid = torchreid.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True)
    reid.eval()

    dummy = torch.zeros(1, 3, height, width, dtype=torch.float32)

    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic else None

    REID_ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = dict(
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
    )
    try:
        # `torch.onnx.export`'s default flipped to the dynamo-based exporter
        # (`dynamo=True`) in recent torch releases (confirmed present, and
        # defaulting to True, in this sandbox's torch==2.13.0), which
        # requires the separate `onnxscript` package (not pinned anywhere in
        # this project) and raises `ModuleNotFoundError: No module named
        # 'onnxscript'` without it. `dynamo=False` forces the older, stable
        # TorchScript-based exporter — the one this function's docstring
        # means by "no built-in exporter" — with no extra dependency.
        torch.onnx.export(reid, dummy, str(REID_ONNX_PATH), dynamo=False, **export_kwargs)
    except TypeError:
        # requirements-export.txt pins torch==2.7.1; if that release (or an
        # older one) doesn't yet accept a `dynamo=` kwarg, fall back to
        # calling without it — whatever exporter is default on that version
        # is, by construction, dependency-free on that version.
        torch.onnx.export(reid, dummy, str(REID_ONNX_PATH), **export_kwargs)
    print(f"[export] reid embedder -> {REID_ONNX_PATH}")
    return REID_ONNX_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", action="store_true", help="export models/yolov10m.pt")
    parser.add_argument("--posecls", action="store_true", help="export models/trained_yolo11m.pt")
    parser.add_argument("--reid", action="store_true", help="export the OSNet Re-ID backbone")
    parser.add_argument("--all", action="store_true", help="export all three models")
    parser.add_argument("--imgsz-detector", type=int, default=640)
    parser.add_argument("--imgsz-posecls", type=int, default=224)
    args = parser.parse_args()

    if not (args.detector or args.posecls or args.reid or args.all):
        parser.error("nothing to export — pass --detector/--posecls/--reid or --all")

    if args.detector or args.all:
        export_detector(imgsz=args.imgsz_detector)
    if args.posecls or args.all:
        export_posecls(imgsz=args.imgsz_posecls)
    if args.reid or args.all:
        export_reid()


if __name__ == "__main__":
    main()
