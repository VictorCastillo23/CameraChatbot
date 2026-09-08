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
hardware with the real exported weights — nothing here has been numerically
validated against the torch path in this sandbox.
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

        blobs, ratios, pads, shapes = [], [], [], []
        for img in img_list:
            lb_img, ratio, pad = self._letterbox(img, (self.imgsz, self.imgsz))
            blobs.append(self._to_blob(lb_img))
            ratios.append(ratio)
            pads.append(pad)
            shapes.append(img.shape[:2])  # (H, W) of the ORIGINAL image

        batch = np.concatenate(blobs, axis=0)  # (N, 3, imgsz, imgsz)
        raw = self.session.run([self.output_name], {self.input_name: batch})[0]

        return self._postprocess(raw, ratios, pads, shapes, conf, iou)

    def _letterbox(self, img, new_shape, color=114, stride=32, auto=False, scaleup=True):
        """Replicates `ultralytics.data.augment.LetterBox` exactly.

        Resize by `min(new_h/h, new_w/w)`, pad with constant `color`, pads
        split evenly (±0.5px handled via the same `round(x ± 0.1)`
        asymmetric rounding Ultralytics uses so the two sides never differ
        by more than one pixel).

        `auto=False` by default — this is NOT an oversight: Ultralytics'
        own predictor (`ultralytics/engine/predictor.py::pre_transform`)
        only sets `auto=True` when `self.model.format == "pt"` (or a
        dynamic-shape non-imx backend); for every exported format,
        including ONNX, it is hard-coded `False` (confirmed by reading the
        installed `ultralytics==8.4.115` source). With `auto=False`, every
        letterboxed image is padded to the FULL `new_shape` square
        (typically a multiple of 32 already, e.g. 640), not just to the
        nearest stride multiple of its own scaled size — this is what
        makes `np.concatenate(blobs, axis=0)` in `__call__` valid across a
        batch of differently-shaped/aspect-ratio source images. Using
        `auto=True` here would produce a different padded canvas size per
        image and silently break batching (or require per-image inference
        calls), and would also not match the fixed spatial shape the
        exported ONNX graph was traced/tested against.

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
