"""ONNX Runtime model loaders — the Fase 3b rollback target, INERT for now.

Sibling of `camerachatbot/runtime/loaders_torch.py`, exposing the identical
three-function contract (`load_detector()`, `load_posecls()`, `load_reid()`,
all zero required args) so a future caller can switch backends by changing
one import line:

    # camerachatbot/runtime/bootstrap.py, Fase 3b (NOT this PR):
    from camerachatbot.runtime.loaders_onnx import load_detector, load_posecls, load_reid
    # ROLLBACK: comment the line above, uncomment the line below
    # from camerachatbot.runtime.loaders_torch import load_detector, load_posecls, load_reid

**Nothing imports this module yet.** `bootstrap.py` is unmodified in this
PR — it still defines and calls its own inline `load_yolo_models()` /
`load_reid(device)` / `get_device()`, using the torch/ultralytics/torchreid
stack exactly as before. Wiring this module in is Fase 3b's job, explicitly
gated on the user running `tools/verify_onnx_parity.py` (this PR also adds
that tool) on real hardware with the real exported ONNX weights.

Expects the ONNX files `tools/export_to_onnx.py` produces:
`models/yolov10m.onnx`, `models/trained_yolo11m.onnx`,
`models/osnet_x1_0.onnx` — none of which are committed to the repo (same
convention as the `.pt`/ONNX-attribute-model files already gitignored under
`models/`).

**Why `onnx_providers()` is a local copy, not imported from `bootstrap.py`**:
`bootstrap.py::onnx_providers()` already exists and does exactly what this
module needs (CUDA provider first if available, else CPU-only). Importing
it directly (`from camerachatbot.runtime.bootstrap import onnx_providers`)
would work today, but would set up a circular import for Fase 3b: once
`bootstrap.py` gains the switch line `from camerachatbot.runtime.loaders_onnx
import ...`, Python would need to fully import `loaders_onnx` — which would
try to import back from the *not-yet-finished-importing* `bootstrap` module.
Whether that resolves depends fragilely on exact statement ordering inside
`bootstrap.py`, which is a landmine not worth setting for a five-line pure
function. This module therefore carries its own copy. If Fase 3c's
requirements-slimming pass wants a single source of truth, the clean fix
then is extracting `onnx_providers()` into a small shared, dependency-free
module (e.g. `camerachatbot/runtime/onnx_utils.py`) that both `bootstrap.py`
and this module import — deliberately out of scope here since it would be
an unnecessary edit to `bootstrap.py` in an otherwise-inert PR.

**`load_reid()`'s return shape intentionally differs from
`loaders_torch.py::load_reid()`**: the torch loader returns
`(reid_model, transform)` because `torchreid`'s model is a bare `nn.Module`
that needs an external preprocessing transform; `OSNetOnnxEmbedder` owns its
own preprocessing internally (see `camerachatbot/detectors/onnx_reid.py`),
so this loader returns just the embedder. This is exactly the "`reid_model`
key's value changes type, `reid_transform` key disappears" asymmetry the
design's own `bootstrap.py` rewiring table documents — Fase 3b's actual
`init_runtime()` edit is where that asymmetry gets absorbed into `RUNTIME`,
not here.
"""

import onnxruntime as ort

from camerachatbot import paths
from camerachatbot.detectors.onnx_yolo import YOLOOnnxDetector
from camerachatbot.detectors.onnx_pose_classifier import PoseClsOnnxClassifier
from camerachatbot.detectors.onnx_reid import OSNetOnnxEmbedder

YOLO_DET_ONNX_PATH = (paths.MODELS_DIR / "yolov10m.onnx").resolve()
YOLO_POSECLS_ONNX_PATH = (paths.MODELS_DIR / "trained_yolo11m.onnx").resolve()
REID_ONNX_PATH = (paths.MODELS_DIR / "osnet_x1_0.onnx").resolve()


def onnx_providers():
    """CUDA provider first if available, else CPU-only.

    Deliberate local copy of `bootstrap.py::onnx_providers()` — see this
    module's docstring for why it is not imported from `bootstrap.py`.
    """
    av = ort.get_available_providers()
    if "CUDAExecutionProvider" in av:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _assert_exists(path, what):
    if not path.exists():
        raise FileNotFoundError(
            f"[CONFIG] No se encontró {what}: {path} — run "
            f"`python tools/export_to_onnx.py --all` first."
        )


def load_detector():
    """Load the ONNX person/object detector (`models/yolov10m.onnx`)."""
    _assert_exists(YOLO_DET_ONNX_PATH, "detector ONNX")
    return YOLOOnnxDetector(YOLO_DET_ONNX_PATH, providers=onnx_providers())


def load_posecls():
    """Load the ONNX sit/stand pose classifier (`models/trained_yolo11m.onnx`)."""
    _assert_exists(YOLO_POSECLS_ONNX_PATH, "pose classifier ONNX")
    return PoseClsOnnxClassifier(YOLO_POSECLS_ONNX_PATH, providers=onnx_providers())


def load_reid():
    """Load the ONNX OSNet Re-ID embedder (`models/osnet_x1_0.onnx`).

    Returns a single `OSNetOnnxEmbedder` — see this module's docstring for
    why that differs in shape from `loaders_torch.py::load_reid()`'s
    `(model, transform)` tuple.
    """
    _assert_exists(REID_ONNX_PATH, "reid embedder ONNX")
    return OSNetOnnxEmbedder(REID_ONNX_PATH, providers=onnx_providers())
