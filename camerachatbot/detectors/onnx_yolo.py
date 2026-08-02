"""ONNX Runtime replacement for the Ultralytics `YOLO` detector.

`YOLOOnnxDetector` duck-types the Ultralytics `Results` iteration shape that
`person_reid.py::detect_and_embed()` already consumes (`self.model(batch_imgs,
verbose=False, conf=..., iou=...)` at the batch call site, plus the
single-image fallback) so the caller needs **zero** changes: `__call__`
always returns a `list[DetResult]`, and each `DetResult.boxes` yields `Box`
elements with `.xyxy` (1,4), `.conf` (1,), `.cls` (1,) — exactly what
`for b in boxes: b.cls[0]; b.xyxy[0]; b.conf[0]` expects today.

Inert module: nothing under `camerachatbot/runtime/` imports this yet.
`bootstrap.py` still loads the torch/ultralytics stack (see
`camerachatbot/runtime/bootstrap.py::load_yolo_models()`). Wiring happens in
Fase 3b, gated on the user running `tools/verify_onnx_parity.py` on real
hardware with the real exported weights.

**PR4b gate-review fix, real weights/images, not theoretical**: an earlier
version of `__call__()` hardcoded `_letterbox(..., auto=False)` for every
call — always padding to the full `imgsz`x`imgsz` square. Ultralytics'
own predictor uses `auto=True` (minimum-rectangle canvas, no wasted square
padding) whenever every image in a call shares one original shape, which
`args.rect` defaults to allowing in predict mode — confirmed empirically,
not assumed (see `__call__()`'s inline comment for the full trace). That
mismatch was the entire root cause of `tools/verify_onnx_parity.py`'s
reported detector conf-diff drift (up to 0.068 on matched boxes, 8
unmatched boxes at 0.25-0.27 confidence across 7 of 14 real test images):
feeding torch's own preprocessed tensor directly into this ONNX graph
produced a bit-identical output (max diff 0.0000), proving it was a
letterbox-canvas mismatch, not backend numeric precision. `__call__()` now
computes `auto` per-call from whether every image in the batch shares one
shape, matching Ultralytics' behavior exactly for the common case
(`person_reid.py`'s real call site batches one camera/video's frames at a
time, which share one resolution) while keeping the full-square fallback
for genuinely mixed-shape batches (required for `np.concatenate`).
"""

import ast

import cv2
import numpy as np
import onnxruntime as ort

# Canonical COCO 80-class names (id -> name), copied verbatim from
# ultralytics/cfg/datasets/coco.yaml so the fallback path (used only when the
# ONNX file carries no `names` metadata) matches `.pt` model behavior exactly.
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}


class Box:
    """Duck-types one element yielded by iterating an Ultralytics `Boxes`."""

    __slots__ = ("xyxy", "conf", "cls")

    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy  # (1, 4) float32, original-image pixel coords
        self.conf = conf  # (1,) float32
        self.cls = cls    # (1,) float32 (holds an int class id)


class DetResult:
    """Duck-types one element of the list Ultralytics returns per call."""

    __slots__ = ("names", "boxes")

    def __init__(self, names, boxes):
        self.names = names        # dict[int, str]
        self.boxes = boxes        # list[Box]


class YOLOOnnxDetector:
    """ONNX Runtime YOLO detector wrapper.

    Handles two output layouts, selected per-inference from the raw output
    shape (never assumed from config, since a mis-detected layout silently
    double-suppresses or floods detections):

    - NMS-free (`yolov10m`, the production detector): `(N, 300, 6)` =
      `[x1, y1, x2, y2, conf, cls]` already decoded in letterboxed-image
      space, one row per candidate, no NMS needed.
    - v8-style (`(N, 4 + nc, num_anchors)`, e.g. `(N, 84, 8400)` for 80 COCO
      classes): box params come first, need transpose to
      `(num_anchors, 4 + nc)`, per-anchor class-confidence argmax, and
      per-class `cv2.dnn.NMSBoxesBatched` (Ultralytics' default is per-class
      NMS, `agnostic_nms=False` — boxes of different classes must not
      suppress each other).
    """

    def __init__(self, onnx_path, providers=None, imgsz=640, conf=0.25, iou=0.45, names=None):
        self.session = ort.InferenceSession(
            str(onnx_path), providers=providers or ["CPUExecutionProvider"]
        )
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.names = names if names is not None else (self._read_names_metadata() or COCO_NAMES)

    def _read_names_metadata(self):
        """Read `names` from the ONNX file's custom metadata, if present.

        `.pt` models carry `model.names` in memory; ONNX has no such
        attribute, so `tools/export_to_onnx.py` writes it into
        `metadata_props` at export time (Ultralytics' own exporter already
        does this via `str(model.names)` — see that tool's docstring for the
        full explanation). Values are written as Python-dict `repr` strings
        (`"{0: 'person', 1: 'bicycle', ...}"`), so `ast.literal_eval` — not
        `json.loads` — is the correct parser (unquoted int keys are not
        valid JSON).
        """
        try:
            meta = self.session.get_modelmeta().custom_metadata_map
            raw = meta.get("names")
            if not raw:
                return None
            parsed = ast.literal_eval(raw)
            return {int(k): v for k, v in parsed.items()}
        except Exception:
            return None

    def __call__(self, imgs, verbose=False, conf=None, iou=None):
        conf = self.conf if conf is None else float(conf)
        iou = self.iou if iou is None else float(iou)

        single = not isinstance(imgs, (list, tuple))
        img_list = [imgs] if single else list(imgs)

        # `auto` — replicates `ultralytics/engine/predictor.py::pre_transform()`'s
        # own `same_shapes` gate exactly: **real, measured discovery, not a
        # theoretical concern** — `tools/verify_onnx_parity.py`'s reported
        # detector conf-diff drift (up to 0.068 on matched pairs, plus 8
        # unmatched high-confidence boxes at 0.25-0.27) was root-caused to
        # THIS module always forcing `auto=False` (full `imgsz`x`imgsz`
        # square canvas), while Ultralytics' own predictor uses `auto=True`
        # (minimum-rectangle canvas, padded only to the nearest `stride`
        # multiple of the scaled dimension) whenever every image in the call
        # shares the same original shape — confirmed empirically: `YOLO(...)
        # .predict(...)`'s own `args.rect` defaults to `True` for predict
        # mode (NOT the `False` the module docstring below previously
        # assumed by reading only `ultralytics/cfg/default.yaml`'s
        # train/val-section default, without checking the predict-mode
        # runtime value), so `same_shapes and args.rect and format=="pt"`
        # reduces to just `same_shapes` in practice. Feeding torch's own
        # rectangular-canvas preprocessed tensor directly into this same
        # ONNX graph (which exports with dynamic height/width — confirmed
        # via `sess.get_inputs()[0].shape == ['batch', 3, 'height',
        # 'width']`) produces a BIT-IDENTICAL `(1, 300, 6)` output to the
        # torch forward pass (max conf diff 0.0000 across all boxes) —
        # proof the drift was 100% a canvas-shape mismatch, not backend
        # numeric precision. `person_reid.py::detect_and_embed()`'s real
        # call site batches images read straight from one `frames_folder`
        # (one camera/video per call), which in practice always share one
        # resolution — exactly the `same_shapes=True` case this fixes.
        #
        # For a genuinely MIXED-shape batch (`same_shapes=False`), `auto`
        # stays `False` (full square canvas) — this is NOT a regression,
        # it is required: `np.concatenate(blobs, axis=0)` below needs every
        # blob in the batch to share one canvas size, and only the full
        # square canvas is shape-independent of the per-image aspect ratio.
        same_shapes = len({img.shape for img in img_list}) == 1
        auto = same_shapes

        blobs, ratios, pads, shapes = [], [], [], []
        for img in img_list:
            lb_img, ratio, pad = self._letterbox(img, (self.imgsz, self.imgsz), auto=auto)
            blobs.append(self._to_blob(lb_img))
            ratios.append(ratio)
            pads.append(pad)
            shapes.append(img.shape[:2])  # (H, W) of the ORIGINAL image

        batch = np.concatenate(blobs, axis=0)  # (N, 3, canvas_h, canvas_w)
        raw = self.session.run([self.output_name], {self.input_name: batch})[0]

        return self._postprocess(raw, ratios, pads, shapes, conf, iou)

    def _letterbox(self, img, new_shape, color=114, stride=32, auto=False, scaleup=True):
        """Replicates `ultralytics.data.augment.LetterBox` exactly.

        Resize by `min(new_h/h, new_w/w)`, pad with constant `color`, pads
        split evenly (±0.5px handled via the same `round(x ± 0.1)`
        asymmetric rounding Ultralytics uses so the two sides never differ
        by more than one pixel).

        `auto=False` is only the METHOD default (used directly if called
        standalone) — `__call__` above always passes an explicit `auto=`
        computed from whether every image in the current call shares one
        original shape, mirroring `ultralytics/engine/predictor.py
        ::pre_transform`'s own `same_shapes and self.args.rect and
        (self.model.format == "pt" or ...)` gate.

        **Correction (PR4b gate review, real weights)**: an earlier version
        of this docstring claimed `auto=False` was ALWAYS correct because
        "every exported format, including ONNX, is hard-coded auto=False"
        in Ultralytics' predictor — that is true for the format check, but
        incomplete: it silently assumed `self.args.rect` defaults to
        `False` (true for `train`/`val` per `ultralytics/cfg/default.yaml`,
        but NOT for `predict` mode — confirmed empirically: a freshly
        loaded `YOLO(...).predict(img)` call has `predictor.args.rect ==
        True`). So for the format this module actually mimics — a `.pt`
        model doing a single-image or same-shape-batch predict call —
        Ultralytics' real, measured `auto` value is `True`, not `False`.
        Hard-coding `auto=False` unconditionally caused a real, measured
        parity gap (`tools/verify_onnx_parity.py`: up to 0.068 conf diff on
        matched boxes, 8 unmatched boxes at 0.25-0.27 confidence) — see
        this module's top docstring for the full root-cause trace.

        `auto=True` (`dw, dh` reduced modulo `stride` instead of padded to
        the full square) DOES still require every image in the batch to
        share the same original shape for `np.concatenate` to stay valid
        (all of them then compute the identical padded canvas size) — that
        is exactly the condition `__call__` checks before choosing `auto`.
        For a genuinely mixed-shape batch, `auto=False` (full square,
        shape-independent of aspect ratio) remains the only option that
        keeps batching valid, and is not a parity gap in that case since
        Ultralytics' own `same_shapes` gate would ALSO force `auto=False`
        for the torch side under the same mixed-shape condition.

        Returns `(padded_img, ratio, (dw, dh))` — `ratio` is a single float
        (uniform scale, aspect ratio preserved) and `(dw, dh)` is the
        *unrounded* half-padding, which is what `_postprocess`'s
        un-letterbox step must subtract before dividing by `ratio` (using
        the rounded pixel padding here would introduce a systematic
        sub-pixel coordinate bias).
        """
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        h, w = img.shape[:2]
        r = min(new_shape[0] / h, new_shape[1] / w)
        if not scaleup:
            r = min(r, 1.0)

        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:
            dw, dh = dw % stride, dh % stride
        dw /= 2
        dh /= 2

        if (w, h) != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color, color, color)
        )
        return img, r, (dw, dh)

    @staticmethod
    def _to_blob(img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return img[None, ...]  # NCHW, N=1

    @staticmethod
    def _unletterbox(boxes_xyxy, ratio, pad, orig_w, orig_h):
        if boxes_xyxy.size == 0:
            return boxes_xyxy
        dw, dh = pad
        boxes_xyxy = boxes_xyxy.copy()
        boxes_xyxy[:, [0, 2]] -= dw
        boxes_xyxy[:, [1, 3]] -= dh
        boxes_xyxy[:, :4] /= ratio
        boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, orig_w)
        boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, orig_h)
        return boxes_xyxy

    @staticmethod
    def _xywh_to_xyxy(box_xywh):
        if box_xywh.size == 0:
            return box_xywh.reshape(0, 4)
        cx, cy, w, h = box_xywh[:, 0], box_xywh[:, 1], box_xywh[:, 2], box_xywh[:, 3]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return np.stack([x1, y1, x2, y2], axis=1)

    def _postprocess(self, raw, ratios, pads, shapes, conf, iou):
        raw = np.asarray(raw)
        n = len(ratios)

        # Layout decision made ONCE, from the whole batch's output shape —
        # this is the single highest-risk branch in this module (design's
        # own warning): getting it wrong silently double-suppresses or
        # floods detections instead of raising.
        nms_free = raw.ndim == 3 and raw.shape[-1] == 6

        if raw.shape[0] != n:
            raise RuntimeError(
                f"YOLOOnnxDetector: ONNX session output batch dimension "
                f"({raw.shape[0]}) does not match the number of input images "
                f"({n}). Silently reusing image-0's detections for every "
                f"image would duplicate/misattribute results across the "
                f"batch instead of failing loudly — check the exported "
                f"model's dynamic-batch axis and the output layout branch "
                f"(nms_free={nms_free}, raw.shape={raw.shape})."
            )

        results = []
        for i in range(n):
            per_image = raw[i]
            orig_h, orig_w = shapes[i]
            ratio_i, pad_i = ratios[i], pads[i]

            if nms_free:
                boxes_xyxy = per_image[:, :4].astype(np.float32)
                scores = per_image[:, 4].astype(np.float32)
                cls_ids = per_image[:, 5].astype(np.float32)

                keep = scores >= conf
                boxes_xyxy = boxes_xyxy[keep]
                scores = scores[keep]
                cls_ids = cls_ids[keep]
                # Model is NMS-free by construction (YOLOv10 head) — no
                # cv2.dnn.NMSBoxes call here, per the design's explicit
                # instruction that NMS runs ONLY in the v8-style branch.
            else:
                per_image = per_image.T  # (4+nc, num_anchors) -> (num_anchors, 4+nc)
                box_xywh = per_image[:, :4].astype(np.float32)
                cls_scores = per_image[:, 4:]

                cls_ids_f = np.argmax(cls_scores, axis=1)
                scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids_f].astype(np.float32)

                keep = scores >= conf
                box_xywh = box_xywh[keep]
                scores = scores[keep]
                cls_ids_f = cls_ids_f[keep]

                boxes_xyxy = self._xywh_to_xyxy(box_xywh)

                if len(boxes_xyxy) > 0:
                    # Per-class NMS: `cv2.dnn.NMSBoxesBatched` suppresses only
                    # within the same `class_ids` group, matching Ultralytics'
                    # own default (`agnostic_nms=False` — boxes of different
                    # classes must never suppress each other, e.g. an
                    # overlapping "person" and "backpack" box are both kept).
                    # A plain `cv2.dnn.NMSBoxes` call here would be
                    # class-agnostic and wrongly drop one of them.
                    nms_input = [
                        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                        for x1, y1, x2, y2 in boxes_xyxy
                    ]
                    kept_idx = cv2.dnn.NMSBoxesBatched(
                        nms_input, scores.tolist(), cls_ids_f.astype(np.int32).tolist(), conf, iou
                    )
                    kept_idx = np.array(kept_idx).reshape(-1) if len(kept_idx) else np.empty(0, dtype=int)
                    boxes_xyxy = boxes_xyxy[kept_idx]
                    scores = scores[kept_idx]
                    cls_ids_f = cls_ids_f[kept_idx]

                cls_ids = cls_ids_f.astype(np.float32)

            boxes_xyxy = self._unletterbox(boxes_xyxy, ratio_i, pad_i, orig_w, orig_h)

            det_boxes = [
                Box(
                    xyxy=np.array([[x1, y1, x2, y2]], dtype=np.float32),
                    conf=np.array([s], dtype=np.float32),
                    cls=np.array([c], dtype=np.float32),
                )
                for (x1, y1, x2, y2), s, c in zip(boxes_xyxy, scores, cls_ids)
            ]
            results.append(DetResult(names=self.names, boxes=det_boxes))

        return results
