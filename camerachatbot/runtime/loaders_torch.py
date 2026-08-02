"""Torch/Ultralytics/torchreid model loaders — today's runtime, unchanged.

This is the CURRENT loading code that `camerachatbot/runtime/bootstrap.py`
still calls today (`load_yolo_models()` and `load_reid(device)`'s bodies,
plus `get_device()`), relocated here **verbatim** (only renamed/split to
match the shared three-function contract `load_detector()` / `load_posecls()`
/ `load_reid()` that `camerachatbot/runtime/loaders_onnx.py` also
implements — see that module's docstring for the rollback mechanism this
pair of sibling modules exists for).

`bootstrap.py` itself is **NOT modified** to import from here in this PR —
it still defines and calls its own `load_yolo_models()`/`load_reid()`/
`get_device()` inline, byte-for-byte the same as before this PR. This
module exists purely as an importable, independently-loadable copy of that
same logic, for two reasons:

1. `tools/verify_onnx_parity.py` needs to import the torch loading path and
   the ONNX loading path *side by side* in the same process — importing
   `bootstrap.py` for the torch side would also trigger its Flask app /
   Supabase client / `.env`-dependent construction, none of which the
   parity tool needs or wants.
2. Fase 3b's actual rollback is then a single import-line edit in
   `bootstrap.py` (see `loaders_onnx.py`'s docstring for that exact diff) —
   this module needs to already exist and be independently correct BEFORE
   that edit lands, not be written at the same time as it.

`load_reid()`'s signature is the one deliberate adaptation from today's
`bootstrap.py::load_reid(device)`: it now resolves its own device via
`get_device()` internally (zero required args), matching
`loaders_onnx.py::load_reid()`'s zero-arg shape so a future caller can
import either module and call all three functions identically. Its
**return shape** still differs from the ONNX sibling on purpose — see
`loaders_onnx.py::load_reid()`'s docstring for why that asymmetry is
correct, not an oversight.
"""

import torch
import torchreid
from ultralytics import YOLO

from camerachatbot import paths

YOLO_DET_PATH = (paths.MODELS_DIR / "yolov10m.pt").resolve()
YOLO_POSECLS_PATH = (paths.MODELS_DIR / "trained_yolo11m.pt").resolve()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _assert_exists(path, what):
    if not path.exists():
        raise FileNotFoundError(f"[CONFIG] No se encontró {what}: {path}")


def load_detector():
    """Load the YOLO person/object detector (`models/yolov10m.pt`).

    Verbatim body from today's `bootstrap.py::load_yolo_models()`, split to
    return only the detector half — `load_posecls()` below is the other
    half of what used to be one combined function.
    """
    _assert_exists(YOLO_DET_PATH, "YOLO detección")
    yolo_det = YOLO(str(YOLO_DET_PATH))
    try:
        yolo_det.fuse()
    except Exception:
        pass
    return yolo_det


def load_posecls():
    """Load the YOLO-CLS sit/stand pose classifier (`models/trained_yolo11m.pt`)."""
    _assert_exists(YOLO_POSECLS_PATH, "YOLO-CLS pose sentado/de pie")
    yolo_posecls = YOLO(str(YOLO_POSECLS_PATH))
    try:
        yolo_posecls.fuse()
    except Exception:
        pass
    return yolo_posecls


def load_reid():
    """Load the OSNet Re-ID model + its torchreid test-time transform.

    Returns `(reid_model, transform)` — a torch `nn.Module` plus a
    torchvision-style callable transform, exactly like today's
    `bootstrap.py::load_reid(device)` (device now resolved internally via
    `get_device()` rather than taken as a parameter, see this module's
    docstring).
    """
    device = get_device()
    reid = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=1000,
        pretrained=True,
    )
    reid.eval().to(device)

    _, transform = torchreid.data.transforms.build_transforms(height=256, width=128)
    return reid, transform
