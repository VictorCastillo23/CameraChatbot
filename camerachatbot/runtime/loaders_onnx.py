"""ONNX Runtime model loaders — the sole model-loading module for the production runtime.

Since Fase 3c, this is the ONLY loader module in the codebase — the former
torch/ultralytics/torchreid sibling (`camerachatbot/runtime/loaders_torch.py`)
was deleted, and `bootstrap.py::init_runtime()` imports its three-function
contract (`load_detector()`, `load_posecls()`, `load_reid()`, all zero
required args) directly and unconditionally:

    # camerachatbot/runtime/bootstrap.py:
    from camerachatbot.runtime.loaders_onnx import load_detector, load_posecls, load_reid

There is no alternate loader path to switch between anymore. Rolling back to
the torch-based runtime would mean reverting to a pre-Fase-3c git commit,
not choosing between two loader modules.

Expects the ONNX files `tools/export_to_onnx.py` produces:
`models/yolov10m.onnx`, `models/trained_yolo11m.onnx`,
`models/osnet_x1_0.onnx` — none of which are committed to the repo (same
convention as the `.pt`/ONNX-attribute-model files already gitignored under
`models/`).

**Why `onnx_providers()` is a local copy, not imported from `bootstrap.py`**:
`bootstrap.py::onnx_providers()` already exists and does exactly what this
module needs (CUDA provider first if available, else CPU-only). Importing
it directly (`from camerachatbot.runtime.bootstrap import onnx_providers`)
would create a circular import: `bootstrap.py` imports `load_detector`/
`load_posecls`/`load_reid` from this module, so this module importing back
from `bootstrap` would require Python to finish importing `bootstrap` before
`bootstrap` itself finishes importing this module. This module therefore
carries its own copy rather than depending on `bootstrap.py`. A future
cleanup could extract `onnx_providers()` into a small shared, dependency-free
module (e.g. `camerachatbot/runtime/onnx_utils.py`) that both `bootstrap.py`
and this module import, removing the duplication without reintroducing the
circular import.
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

    Returns a single `OSNetOnnxEmbedder`, which owns its own preprocessing
    internally (see `camerachatbot/detectors/onnx_reid.py`) — there is no
    separate transform object to return alongside it.
    """
    _assert_exists(REID_ONNX_PATH, "reid embedder ONNX")
    return OSNetOnnxEmbedder(REID_ONNX_PATH, providers=onnx_providers())
