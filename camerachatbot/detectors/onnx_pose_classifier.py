"""ONNX Runtime replacement for the Ultralytics YOLO-CLS sit/stand classifier.

**Not a thin wrapper.** `pose_action_classifier.py::PoseActionClassifier` today
outsources batching, precision and preprocessing entirely to Ultralytics
(`self.model(crops, imgsz=224, batch=16, half=torch.cuda.is_available(),
verbose=False)`, then reads `results[i].probs`). None of that machinery
exists once the model is exported to ONNX, so `PoseClsOnnxClassifier`
reimplements preprocessing, batching and the argmax/confidence readout by
hand.

Inert module: `PoseActionClassifier.run_on_json()` is **not** modified in
this PR — this file only creates the standalone, independently-testable
`PoseClsOnnxClassifier` class. Wiring it into `bootstrap.py`/
`pose_action_classifier.py` is Fase 3b's job, gated on the user running
`tools/verify_onnx_parity.py` on real hardware with the real exported
weights.

Three hazards this module must get right (from the design doc, all
confirmed — not assumed — by reading the installed `ultralytics==8.4.115`
source in this sandbox's venv):

1. **Preprocessing parity is NOT a naive resize-to-224x224.**
   `ultralytics/models/yolo/classify/predict.py::ClassificationPredictor
   .setup_source()` picks the transform pipeline: for a `.pt` model whose
   embedded `imgsz` still matches the requested one, it reuses
   `self.model.model.transforms` — the *exact* `torch_transforms` object
   `ultralytics/data/dataset.py::ClassificationDataset.__init__` built at
   training/validation time via `classify_transforms(size=args.imgsz)`
   (no `mean`/`std` override). For any other case (ONNX, changed imgsz) it
   calls `classify_transforms(self.imgsz)` directly — same function, same
   defaults. Either way the pipeline (`ultralytics/data/augment.py
   ::classify_transforms`) is: `torchvision.transforms.Resize(size)`
   (scalar `size` -> shortest-edge resize, aspect ratio preserved,
   bilinear) -> `CenterCrop(size)` -> `ToTensor()` (uint8 [0,255] ->
   float32 [0,1]) -> `Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)`.
   A naive resize-to-224x224 (ignoring aspect ratio) changes pixel content
   at the crop's edges/corners and *will* fail the ±0.02 probability
   parity tolerance in `tools/verify_onnx_parity.py`.

5. **The resize step operates on a PIL Image, not a torch Tensor —
   confirmed by reading `ClassificationPredictor.preprocess()` directly**:
   `self.transforms(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))`.
   `T.Resize`'s `forward()` dispatches on the input type; for a PIL input
   it delegates straight to `torchvision.transforms._functional_pil.resize()`,
   which is nothing more than `img.resize(size, Image.BILINEAR)` — Pillow's
   own C resize, not torchvision's tensor-path resize. This matters because
   Pillow's BILINEAR resize *always* antialiases internally when
   downsampling (there is no separate antialias flag on the PIL codepath —
   `torchvision.transforms.functional.resize()`'s own docstring says so
   explicitly: "on PIL images, antialiasing is always applied on bilinear
   or bicubic modes"). This is a **third, distinct filter** from both
   `cv2.INTER_AREA` (a box filter) and torch's own tensor-path antialiased
   bilinear (a separable triangle filter applied in float, pre-scaled by
   the resize ratio) — approximating it with `cv2.INTER_AREA` (as an
   earlier revision of this module did) leaves a real, measurable residual
   on aggressive downsamples (confirmed: up to 0.0645 max probability
   diff on 2/14 real `keyFrames` images in `tools/verify_onnx_parity.py`).
   `_preprocess()` below uses `PIL.Image.resize(..., Image.BILINEAR)`
   directly instead, matching the real pipeline's resize call bit-for-bit
   rather than approximating it.

2. **A real correction to the design's own stated hazard**: the design
   assumed `classify_transforms` applies "ImageNet normalize". Reading
   `ultralytics/data/augment.py` directly shows `classify_transforms`'s
   `mean`/`std` parameters default to `DEFAULT_MEAN = (0.0, 0.0, 0.0)` /
   `DEFAULT_STD = (1.0, 1.0, 1.0)` (module-level constants in that same
   file), and neither `ClassificationDataset` (training/val, no `augment`)
   nor `ClassificationPredictor` (inference fallback) ever passes a
   non-default `mean`/`std`. `Normalize(mean=0, std=1)` is the identity
   function. So in this installed ultralytics version, YOLO-CLS
   preprocessing is Resize -> CenterCrop -> `/255` and **nothing else** —
   no ImageNet mean/std subtraction. `_preprocess()` below matches that
   confirmed behavior; it does NOT apply ImageNet normalization. If a
   future ultralytics upgrade changes `DEFAULT_MEAN`/`DEFAULT_STD`, this
   comment (and `tools/verify_onnx_parity.py`'s pose-classifier check)
   is what will catch the drift.

3. **Precision.** `half=torch.cuda.is_available()` disappears entirely —
   ONNX precision is fixed at export time (`export_posecls()` in
   `tools/export_to_onnx.py` exports fp32). This module always runs fp32.

4. **Softmax — do NOT apply it again.** `ultralytics/nn/modules/head.py
   ::Classify.forward()`: `y = x.softmax(1); return y if self.export else
   (y, x)`. The `Classify` head applies softmax internally in BOTH export
   and non-export mode, and `Exporter.export_onnx` sets `m.export = True`
   on every head module before tracing — so the ONNX graph's single output
   is already the softmax'd `(B, num_classes)` probability vector. Applying
   `softmax` a second time in `predict()` would compress every confidence
   value below `conf_threshold=0.60` used by
   `PoseActionClassifier.run_on_json()`, silently breaking the pose gate.
   `predict()` below takes `argmax`/`max` directly off the raw ONNX output.
"""

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


class PoseClsOnnxClassifier:
    """ONNX Runtime replacement for the `trained_yolo11m.pt` YOLO-CLS model.

    `predict(crops)` mirrors what `pose_action_classifier.py` extracts from
    Ultralytics' `results[i].probs` today: a `(pred_idx, conf)` pair per
    crop, in the same order as the input list.
    """

    def __init__(self, onnx_path, providers=None, imgsz=224, batch=16, names=None):
        self.session = ort.InferenceSession(
            str(onnx_path), providers=providers or ["CPUExecutionProvider"]
        )
        self.imgsz = int(imgsz)
        self.batch = int(batch)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # `.pt` carries `model.names` in memory; ONNX has no such attribute.
        # `tools/export_to_onnx.py::export_posecls()` writes it into
        # `metadata_props` the same way `onnx_yolo.py` does for the
        # detector. Callers may also pass `names` explicitly (e.g. a test
        # fixture with no real export step).
        self.names = names if names is not None else (self._read_names_metadata() or {})

    def _read_names_metadata(self):
        """Read `names` from the ONNX file's custom metadata, if present.

        Same `ast.literal_eval`-over-`repr` contract as
        `onnx_yolo.py::YOLOOnnxDetector._read_names_metadata()` — Ultralytics'
        exporter writes `str(model.names)`, a Python-dict repr, not JSON.
        """
        import ast

        try:
            meta = self.session.get_modelmeta().custom_metadata_map
            raw = meta.get("names")
            if not raw:
                return None
            parsed = ast.literal_eval(raw)
            return {int(k): v for k, v in parsed.items()}
        except Exception:
            return None

    def _preprocess(self, crops: list) -> np.ndarray:
        """BGR crops -> `(B, 3, imgsz, imgsz)` float32 NCHW, `classify_transforms` parity.

        Per-crop: BGR->RGB -> `PIL.Image.fromarray` -> resize shortest edge
        to `imgsz` via `Image.resize(..., Image.BILINEAR)` (aspect ratio
        preserved, `int()`-truncated target long-edge exactly as
        `torchvision.transforms.functional._compute_resized_output_size`
        computes it) -> center crop to `(imgsz, imgsz)` (offsets via
        `round((dim - imgsz) / 2.0)`, exactly as
        `torchvision.transforms.functional.center_crop`) -> `/255`.

        No ImageNet mean/std subtraction — see hazard #2 in this module's
        docstring: `classify_transforms`'s `Normalize` is `mean=0, std=1`
        (identity) in this installed ultralytics version.
        """
        n = len(crops)
        batch = np.zeros((n, 3, self.imgsz, self.imgsz), dtype=np.float32)

        for i, crop in enumerate(crops):
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]

            # Resize: shortest edge -> self.imgsz, aspect ratio preserved.
            # Matches torchvision's `int(requested_new_short * long / short)`
            # (truncation, not rounding) for the long edge.
            if w <= h:
                short, long = w, h
                new_short = self.imgsz
                new_long = int(self.imgsz * long / short)
                new_w, new_h = new_short, new_long
            else:
                short, long = h, w
                new_short = self.imgsz
                new_long = int(self.imgsz * long / short)
                new_w, new_h = new_long, new_short

            new_w = max(1, new_w)
            new_h = max(1, new_h)

            # `classify_transforms` runs its `Resize` on a `PIL.Image`, not a
            # tensor (see hazard #5 above) — `T.Resize`'s PIL codepath is
            # exactly `img.resize(size, Image.BILINEAR)`, Pillow's own C
            # resize, which always antialiases BILINEAR/BICUBIC downsamples
            # internally (no separate antialias flag exists on that path).
            # Use PIL directly here instead of approximating with an OpenCV
            # filter, so this matches the real pipeline's resize call
            # bit-for-bit rather than through a different filter kernel.
            pil_img = Image.fromarray(rgb)
            resized_pil = pil_img.resize((new_w, new_h), Image.BILINEAR)
            resized = np.asarray(resized_pil)

            # Center crop to (imgsz, imgsz); pad with 0 first if the resized
            # image is smaller than imgsz along any edge (defensive — should
            # not happen since the short edge is always exactly self.imgsz
            # and the long edge is >= self.imgsz by construction above).
            if new_h < self.imgsz or new_w < self.imgsz:
                padded = np.zeros(
                    (max(new_h, self.imgsz), max(new_w, self.imgsz), 3), dtype=resized.dtype
                )
                padded[:new_h, :new_w] = resized
                resized = padded
                new_h, new_w = resized.shape[:2]

            top = int(round((new_h - self.imgsz) / 2.0))
            left = int(round((new_w - self.imgsz) / 2.0))
            cropped = resized[top:top + self.imgsz, left:left + self.imgsz]

            arr = cropped.astype(np.float32) / 255.0
            batch[i] = np.transpose(arr, (2, 0, 1))  # HWC -> CHW

        return batch

    def predict(self, crops: list) -> list:
        """Predict `(pred_idx, conf)` per crop, batched in chunks of `self.batch`.

        The ONNX output is already softmax'd (see hazard #4 in this
        module's docstring) — no softmax is applied here.
        """
        n = len(crops)
        if n == 0:
            return []

        preds = [None] * n
        for start in range(0, n, self.batch):
            chunk = crops[start:start + self.batch]
            blob = self._preprocess(chunk)
            out = self.session.run([self.output_name], {self.input_name: blob})[0]
            out = np.asarray(out, dtype=np.float32)

            pred_idx = np.argmax(out, axis=1)
            for j, idx in enumerate(pred_idx):
                conf = float(out[j, idx])
                preds[start + j] = (int(idx), conf)

        return preds
