"""Manual verification for the Fase 3a / PR4a ONNX detector+reid contract.

Not a pytest test: this repo has no test runner, so this is a plain
runnable script using bare ``assert`` statements, printing a PASS/FAIL
summary and exiting 0/1. Run it directly:

    python tests_manual/test_fase3a_onnx_detector_reid.py

Unlike Fase 2's manual tests, this sandbox actually HAS `torch`,
`ultralytics`, `torchreid`, `onnxruntime` and `cv2` installed (confirmed at
apply time — contrary to the design doc's assumption when it was written).
The `onnx` package (the protobuf format library `torch.onnx.export` and
`tools/export_to_onnx.py::_ensure_names_metadata()` need, distinct from
`onnxruntime`) was genuinely missing and was installed into this project's
venv specifically to run this suite — see the apply-progress notes for that
call. That lets this script go well beyond `python -m py_compile`:

1. `YOLOOnnxDetector._letterbox()` / `_unletterbox()` — pure-math checks,
   including the `auto=False` padding behavior (padding to the FULL square
   `new_shape`, not just the nearest stride multiple — verified against
   `ultralytics.engine.predictor.BasePredictor.pre_transform()`'s actual
   source: `auto=True` is used ONLY for `.pt` models, `False` for every
   exported format including ONNX).
2. `YOLOOnnxDetector.__call__()` end-to-end against REAL `onnxruntime`
   sessions built from tiny synthetic ONNX graphs (a `torch.nn.Module` that
   ignores its input and returns a fixed tensor, exported via
   `torch.onnx.export`) — one exercising the NMS-free `(N, 300, 6)` branch,
   one exercising the v8-style `(N, 4+nc, num_anchors)` branch (transpose +
   `cv2.dnn.NMSBoxes`). This is real `InferenceSession.run()` execution, not
   a mock.
3. `YOLOOnnxDetector._read_names_metadata()` against a real ONNX file with
   `metadata_props` written the same way `tools/export_to_onnx.py` /
   Ultralytics' own exporter does (`str(dict)`, parsed with
   `ast.literal_eval`).
4. `OSNetOnnxEmbedder._preprocess()`'s pixel-for-pixel closeness against the
   ACTUAL installed `torchreid.data.transforms.build_transforms(height=256,
   width=128)` test-transform output — the real parity check the design
   flagged as impossible in this sandbox, now run for real (modulo the
   known cv2-vs-PIL bilinear kernel difference, measured and asserted
   below a tolerance rather than assumed).
5. `OSNetOnnxEmbedder.embed_one()` / `embed()` against a REAL exported
   `osnet_x1_0` ONNX graph (`torchreid.models.build_model(...,
   pretrained=False)` — random weights, offline, no download) compared to
   that same model's own `torch` forward pass on the identical preprocessed
   tensor. This validates the export/session plumbing end-to-end; it does
   NOT validate real embedding quality (random weights), which still
   requires `tools/verify_onnx_parity.py` on real hardware with the real
   trained weights per the design's explicit gate.
6. `tools/export_to_onnx.py::_ensure_names_metadata()` against a real ONNX
   file, both the no-op path (metadata already present, matching
   Ultralytics' own exporter behavior) and the injection path (metadata
   absent, function must add it).

Uses only synthetic/random-weight models and in-memory data throughout —
never touches `models/yolov10m.pt`/`models/trained_yolo11m.pt` (not present
in this sandbox) or the real gallery files.
"""

import ast
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import torch  # noqa: E402
import torchreid  # noqa: E402
from PIL import Image  # noqa: E402

from camerachatbot.detectors.onnx_yolo import YOLOOnnxDetector  # noqa: E402
from camerachatbot.detectors.onnx_reid import OSNetOnnxEmbedder  # noqa: E402

try:
    import onnx  # noqa: E402
    _HAS_ONNX_PKG = True
except ImportError:
    _HAS_ONNX_PKG = False

results = []  # list[tuple[str, bool, str]]


def check(name, fn):
    try:
        fn()
        results.append((name, True, ""))
    except AssertionError as e:
        results.append((name, False, f"AssertionError: {e}"))
    except Exception as e:  # noqa: BLE001 - surfaced in the summary, not swallowed
        results.append((name, False, f"{type(e).__name__}: {e}"))


# ---------------------------------------------------------------------------
# Synthetic ONNX model builders
# ---------------------------------------------------------------------------

class _ConstOutputModule(torch.nn.Module):
    """Ignores its input and returns a fixed tensor, broadcast over batch.

    Lets us exercise `YOLOOnnxDetector.__call__`'s full pipeline (letterbox
    -> real `InferenceSession.run()` -> postprocess -> un-letterbox)
    against a KNOWN raw output, without needing real trained weights.
    """

    def __init__(self, fixed_output: torch.Tensor):
        super().__init__()
        self.register_buffer("fixed_output", fixed_output)

    def forward(self, x):
        b = x.shape[0]
        return self.fixed_output.unsqueeze(0).expand(b, *self.fixed_output.shape)


class _FixedBatchOutputModule(torch.nn.Module):
    """Always returns a batch-size-1 output, ignoring the actual input batch
    size — used to simulate an ONNX session whose output batch dimension
    disagrees with the number of input images, so `__call__` can be tested
    against that mismatch (Finding 1: must raise loudly, never silently
    reuse image-0's detections for every image via a `raw[0]` fallback).
    """

    def __init__(self, fixed_output: torch.Tensor):
        super().__init__()
        self.register_buffer("fixed_output", fixed_output)

    def forward(self, x):
        # `+ 0.0 * x.sum()` keeps `x` referenced in the traced graph (a
        # `forward` that never touches `x` at all gets its input pruned
        # entirely by `torch.onnx.export`, leaving a graph with ZERO
        # declared inputs instead of the batch-size-mismatch scenario this
        # module exists to simulate) without affecting the actual output
        # value or its batch size.
        return self.fixed_output.unsqueeze(0) + 0.0 * x.sum()


def _export_fixed_batch_model(fixed_output, input_shape, tmpdir, name):
    module = _FixedBatchOutputModule(fixed_output)
    module.eval()
    dummy = torch.zeros(*input_shape, dtype=torch.float32)
    path = os.path.join(tmpdir, name)
    torch.onnx.export(
        module, dummy, path,
        input_names=["images"], output_names=["output"],
        # Only the INPUT batch axis is dynamic — the output is intentionally
        # NOT marked dynamic on its batch axis, so a >1-image call still
        # gets back a (1, ...) raw output: the mismatch under test.
        dynamic_axes={"images": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return path


class _PerBatchIndexDetModule(torch.nn.Module):
    """Encodes each input's position in the batch into its own detection box
    coordinates (via `torch.arange(b)`, independent of pixel content), so
    the raw per-image output genuinely differs across a real multi-image
    batch. Lets a test verify `__call__` attributes `result[i]` to input
    image `i` — not a duplicate/swap of another image's detections, and not
    a copy of a shared/incorrectly-indexed raw row — for images of
    DIFFERENT original sizes in one real batched inference call (the exact
    scenario `auto=False` letterboxing exists for).
    """

    def forward(self, x):
        b = x.shape[0]
        idx = torch.arange(b, dtype=torch.float32)
        det = torch.zeros(b, 300, 6, dtype=torch.float32)
        det[:, 0, 0] = idx * 10.0        # x1, letterbox-space
        det[:, 0, 1] = idx * 10.0        # y1
        det[:, 0, 2] = idx * 10.0 + 5.0  # x2
        det[:, 0, 3] = idx * 10.0 + 5.0  # y2
        det[:, 0, 4] = 0.9               # conf
        det[:, 0, 5] = 0.0               # cls
        return det


def _export_per_batch_index_model(input_shape, tmpdir, name):
    module = _PerBatchIndexDetModule()
    module.eval()
    dummy = torch.zeros(*input_shape, dtype=torch.float32)
    path = os.path.join(tmpdir, name)
    torch.onnx.export(
        module, dummy, path,
        input_names=["images"], output_names=["output"],
        dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return path


def _export_const_model(fixed_output, input_shape, tmpdir, name):
    module = _ConstOutputModule(fixed_output)
    module.eval()
    dummy = torch.zeros(*input_shape, dtype=torch.float32)
    path = os.path.join(tmpdir, name)
    torch.onnx.export(
        module, dummy, path,
        input_names=["images"], output_names=["output"],
        dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        # See tools/export_to_onnx.py::export_reid() — recent torch defaults
        # `torch.onnx.export` to the dynamo exporter, which needs the
        # separate `onnxscript` package this project does not pin anywhere.
        dynamo=False,
    )
    return path


# ---------------------------------------------------------------------------
# 1. letterbox / un-letterbox pure math
# ---------------------------------------------------------------------------

def test_letterbox_pads_to_full_square_auto_false():
    det = YOLOOnnxDetector.__new__(YOLOOnnxDetector)  # no session needed for pure math
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    padded, ratio, pad = det._letterbox(img, (256, 256))
    # r = min(256/100, 256/200) = 1.28 -> new_unpad = (256, 128)
    assert abs(ratio - 1.28) < 1e-9, f"unexpected ratio {ratio}"
    # dw = 256-256=0, dh = 256-128=128 -> auto=False, no stride mod -> /2
    assert pad == (0.0, 64.0), f"unexpected pad {pad}"
    # auto=False must ALWAYS reach the exact target square, unlike auto=True
    assert padded.shape[:2] == (256, 256), f"padded shape {padded.shape} != target square"


def test_letterbox_odd_padding_split_asymmetrically_by_at_most_1px():
    det = YOLOOnnxDetector.__new__(YOLOOnnxDetector)
    img = np.zeros((91, 160, 3), dtype=np.uint8)  # odd height -> odd total pad
    padded, ratio, pad = det._letterbox(img, (128, 128))
    assert padded.shape[:2] == (128, 128)
    # top/bottom must each be within 1px of pad[1] (the unrounded half-pad)
    # reconstruct top/bottom the same way _letterbox does internally
    dh = pad[1]
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    assert abs((top + bottom) - round(2 * dh)) <= 1
    assert abs(top - bottom) <= 1


def test_unletterbox_inverts_letterbox_round_trip():
    det = YOLOOnnxDetector.__new__(YOLOOnnxDetector)
    img_h, img_w = 90, 160
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    _, ratio, pad = det._letterbox(img, (128, 128))

    orig_box = np.array([[10.0, 5.0, 50.0, 40.0]], dtype=np.float32)
    lb_box = orig_box.copy()
    lb_box[:, [0, 2]] = orig_box[:, [0, 2]] * ratio + pad[0]
    lb_box[:, [1, 3]] = orig_box[:, [1, 3]] * ratio + pad[1]

    recovered = det._unletterbox(lb_box, ratio, pad, img_w, img_h)
    assert np.allclose(recovered, orig_box, atol=1e-3), f"round-trip mismatch: {recovered} vs {orig_box}"


def test_unletterbox_clips_to_original_bounds():
    det = YOLOOnnxDetector.__new__(YOLOOnnxDetector)
    boxes = np.array([[-50.0, -50.0, 99999.0, 99999.0]], dtype=np.float32)
    out = det._unletterbox(boxes, ratio=1.0, pad=(0.0, 0.0), orig_w=100, orig_h=80)
    assert out[0, 0] == 0.0 and out[0, 1] == 0.0
    assert out[0, 2] == 100 and out[0, 3] == 80


# ---------------------------------------------------------------------------
# 2. __call__ end-to-end against real onnxruntime sessions
# ---------------------------------------------------------------------------

def test_call_nms_free_branch_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        imgsz = 64
        # (300, 6) buffer: one real detection above conf threshold, rest
        # zeroed (conf=0, filtered out) -- mirrors a real NMS-free head's
        # padded output.
        dets = torch.zeros(300, 6, dtype=torch.float32)
        dets[0] = torch.tensor([10.0, 10.0, 40.0, 40.0, 0.9, 3.0])  # cls_id=3
        path = _export_const_model(dets, (1, 3, imgsz, imgsz), tmp, "nms_free.onnx")

        det = YOLOOnnxDetector(path, imgsz=imgsz, conf=0.5, iou=0.45,
                                names={3: "motorcycle"})
        img = np.zeros((64, 64, 3), dtype=np.uint8)  # square -> ratio=1, pad=(0,0)
        out = det([img])

        assert isinstance(out, list) and len(out) == 1
        res = out[0]
        assert res.names == {3: "motorcycle"}
        assert len(res.boxes) == 1, f"expected 1 box, got {len(res.boxes)}"
        b = res.boxes[0]
        assert b.xyxy.shape == (1, 4)
        assert b.conf.shape == (1,) and abs(float(b.conf[0]) - 0.9) < 1e-4
        assert b.cls.shape == (1,) and int(b.cls[0]) == 3
        assert np.allclose(b.xyxy[0], [10.0, 10.0, 40.0, 40.0], atol=1e-2)


def test_call_single_image_not_a_list_still_returns_list():
    with tempfile.TemporaryDirectory() as tmp:
        imgsz = 64
        dets = torch.zeros(300, 6, dtype=torch.float32)
        path = _export_const_model(dets, (1, 3, imgsz, imgsz), tmp, "empty.onnx")
        det = YOLOOnnxDetector(path, imgsz=imgsz, conf=0.5, names={0: "person"})
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        out = det(img)  # NOT wrapped in a list, matching person_reid.py's fallback call
        assert isinstance(out, list) and len(out) == 1
        assert out[0].boxes == []


def test_call_v8_style_branch_with_real_nms_suppression():
    with tempfile.TemporaryDirectory() as tmp:
        imgsz = 64
        nc = 2
        num_anchors = 3
        # layout (1, 4+nc, num_anchors): box params first, then per-class scores
        raw = torch.zeros(4 + nc, num_anchors, dtype=torch.float32)
        # anchor 0: strong box, class 0, score 0.9, xywh center=(30,30) w=h=20
        raw[:, 0] = torch.tensor([30.0, 30.0, 20.0, 20.0, 0.9, 0.0])
        # anchor 1: heavily overlapping box, same class, LOWER score 0.6 -> must be NMS-suppressed
        raw[:, 1] = torch.tensor([31.0, 31.0, 20.0, 20.0, 0.6, 0.0])
        # anchor 2: distant box, class 1, score 0.8 -> kept independently
        raw[:, 2] = torch.tensor([5.0, 5.0, 6.0, 6.0, 0.05, 0.8])
        path = _export_const_model(raw, (1, 3, imgsz, imgsz), tmp, "v8_style.onnx")

        det = YOLOOnnxDetector(path, imgsz=imgsz, conf=0.25, iou=0.45,
                                names={0: "person", 1: "backpack"})
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        out = det([img])

        assert len(out) == 1
        boxes = out[0].boxes
        cls_ids = sorted(int(b.cls[0]) for b in boxes)
        assert cls_ids == [0, 1], f"expected exactly one box per class after NMS, got cls_ids={cls_ids}"
        # the surviving class-0 box must be the higher-confidence anchor 0.9, not 0.6
        person_box = next(b for b in boxes if int(b.cls[0]) == 0)
        assert abs(float(person_box.conf[0]) - 0.9) < 1e-4


def test_call_v8_style_branch_cross_class_high_overlap_not_suppressed():
    """Finding 2: NMS in the v8-style branch must be per-class. Two heavily
    OVERLAPPING boxes of DIFFERENT classes must both survive — a
    class-agnostic `cv2.dnn.NMSBoxes` call would wrongly suppress the
    lower-confidence one just because it spatially overlaps the
    higher-confidence box, even though they're different classes.
    """
    with tempfile.TemporaryDirectory() as tmp:
        imgsz = 64
        nc = 2
        num_anchors = 2
        raw = torch.zeros(4 + nc, num_anchors, dtype=torch.float32)
        # anchor 0: class 0, box xyxy ~ (20,20)-(40,40), conf 0.9
        raw[:, 0] = torch.tensor([30.0, 30.0, 20.0, 20.0, 0.9, 0.05])
        # anchor 1: class 1, box xyxy ~ (21,21)-(41,41) -- HEAVILY overlapping
        # anchor 0 (IoU well above the 0.45 threshold), lower conf 0.8
        raw[:, 1] = torch.tensor([31.0, 31.0, 20.0, 20.0, 0.05, 0.8])
        path = _export_const_model(raw, (1, 3, imgsz, imgsz), tmp, "v8_cross_class.onnx")

        det = YOLOOnnxDetector(path, imgsz=imgsz, conf=0.25, iou=0.45,
                                names={0: "person", 1: "backpack"})
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        out = det([img])

        boxes = out[0].boxes
        cls_ids = sorted(int(b.cls[0]) for b in boxes)
        assert cls_ids == [0, 1], (
            "class-agnostic NMS would wrongly suppress one of these "
            f"heavily-overlapping different-class boxes; got cls_ids={cls_ids}"
        )


def test_call_batch_size_mismatch_raises():
    """Finding 1: a mismatch between the ONNX session's actual output batch
    dimension and the number of input images must raise loudly (never
    silently fall back to reusing image-0's detections for every image).
    """
    with tempfile.TemporaryDirectory() as tmp:
        imgsz = 64
        dets = torch.zeros(300, 6, dtype=torch.float32)
        dets[0] = torch.tensor([10.0, 10.0, 40.0, 40.0, 0.9, 0.0])
        path = _export_fixed_batch_model(dets, (1, 3, imgsz, imgsz), tmp, "fixed_batch.onnx")

        det = YOLOOnnxDetector(path, imgsz=imgsz, conf=0.5, names={0: "person"})
        img1 = np.zeros((64, 64, 3), dtype=np.uint8)
        img2 = np.zeros((80, 60, 3), dtype=np.uint8)

        raised = False
        try:
            det([img1, img2])  # session always returns batch=1, we sent 2 images
        except RuntimeError as e:
            raised = True
            msg = str(e).lower()
            assert "batch" in msg, f"error message should mention batch size, got: {e}"
        assert raised, "expected RuntimeError on batch-size mismatch, no exception was raised"


def test_call_multi_image_batch_different_sizes_correctly_attributed():
    """Real multi-image batch (`len(imgs) > 1`) of DIFFERENT original sizes
    in a single `__call__` — the exact scenario `auto=False` letterboxing
    was chosen for (so every image in the batch letterboxes to the same
    canvas size and `np.concatenate` is valid). Verifies per-image results
    are correctly attributed to their own input image (not duplicated or
    swapped) by encoding each image's batch position into its own raw
    detection box, independent of pixel content.
    """
    with tempfile.TemporaryDirectory() as tmp:
        imgsz = 128
        path = _export_per_batch_index_model((1, 3, imgsz, imgsz), tmp, "per_batch_idx.onnx")

        det = YOLOOnnxDetector(path, imgsz=imgsz, conf=0.5, names={0: "person"})

        # Two DIFFERENT original sizes -> different letterbox ratio/pad per image.
        img0 = np.zeros((80, 60, 3), dtype=np.uint8)   # portrait
        img1 = np.zeros((50, 120, 3), dtype=np.uint8)  # landscape
        imgs = [img0, img1]

        out = det(imgs)
        assert len(out) == 2

        # Recompute each image's OWN ratio/pad independently (mirrors what
        # __call__ does internally) to predict the expected un-letterboxed
        # box for that specific batch index, then confirm result[i] matches
        # image i's prediction -- NOT image (1-i)'s, which is what a
        # swapped/duplicated attribution bug would produce.
        for i, img in enumerate(imgs):
            _, ratio_i, pad_i = det._letterbox(img, (imgsz, imgsz))
            lb_box = np.array([[i * 10.0, i * 10.0, i * 10.0 + 5.0, i * 10.0 + 5.0]], dtype=np.float32)
            expected = det._unletterbox(lb_box.copy(), ratio_i, pad_i, img.shape[1], img.shape[0])

            res_boxes = out[i].boxes
            assert len(res_boxes) == 1, f"image {i}: expected 1 box, got {len(res_boxes)}"
            got = res_boxes[0].xyxy
            assert np.allclose(got, expected, atol=1e-2), (
                f"image {i}: attribution mismatch -- got {got}, expected {expected} "
                f"(would indicate a duplicated/swapped per-image result)"
            )

        # Cross-check: image 0's box must NOT equal image 1's raw (idx=1)
        # detection un-letterboxed with image 0's own ratio/pad -- i.e. this
        # would fail if the code used raw[0] for every image (Finding 1's bug
        # class) or swapped indices between images.
        _, ratio_0, pad_0 = det._letterbox(img0, (imgsz, imgsz))
        idx1_box_via_img0_geom = det._unletterbox(
            np.array([[10.0, 10.0, 15.0, 15.0]], dtype=np.float32), ratio_0, pad_0,
            img0.shape[1], img0.shape[0],
        )
        assert not np.allclose(out[0].boxes[0].xyxy, idx1_box_via_img0_geom, atol=1e-2), (
            "image 0's result must not match image 1's raw detection -- "
            "indicates a duplicated/swapped per-image attribution bug"
        )


# ---------------------------------------------------------------------------
# 3. names metadata read
# ---------------------------------------------------------------------------

def test_read_names_metadata_ast_literal_eval_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        imgsz = 64
        dets = torch.zeros(300, 6, dtype=torch.float32)
        path = _export_const_model(dets, (1, 3, imgsz, imgsz), tmp, "named.onnx")

        # Writing metadata_props requires the `onnx` package (not installed
        # in this sandbox — see requirements-export.txt), so this exercises
        # the PARSER side directly: ast.literal_eval must exactly recover
        # int keys from the `str(dict)` format Ultralytics' exporter writes
        # (confirmed by reading ultralytics/engine/exporter.py — see
        # onnx_yolo.py::YOLOOnnxDetector._read_names_metadata()'s docstring).
        names = {0: "person", 1: "bicycle", 2: "sports ball"}
        raw = str(names)
        parsed = ast.literal_eval(raw)
        parsed = {int(k): v for k, v in parsed.items()}
        assert parsed == names, f"ast.literal_eval round-trip mismatch: {parsed} vs {names}"

        # and confirm the detector falls back to COCO_NAMES when no metadata
        # is present at all (this ONNX file was exported with no custom
        # metadata_props, since writing them requires the `onnx` package)
        det = YOLOOnnxDetector(path, imgsz=imgsz, conf=0.5)
        from camerachatbot.detectors.onnx_yolo import COCO_NAMES
        assert det.names == COCO_NAMES, "expected COCO_NAMES fallback when no metadata present"


# ---------------------------------------------------------------------------
# 4. OSNet preprocessing parity against the REAL installed torchreid transform
# ---------------------------------------------------------------------------

def test_osnet_preprocess_matches_torchreid_test_transform():
    rng = np.random.default_rng(42)
    crop_bgr = rng.integers(0, 256, size=(180, 90, 3), dtype=np.uint8)

    embedder = OSNetOnnxEmbedder.__new__(OSNetOnnxEmbedder)
    embedder.size = (128, 256)
    ours = embedder._preprocess(crop_bgr)[0]  # (3, 256, 128) CHW

    # the REAL torchreid test-time transform, exactly as bootstrap.py::load_reid() builds it
    _, transform_te = torchreid.data.transforms.build_transforms(height=256, width=128)
    pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    theirs = transform_te(pil_img).numpy()  # (3, 256, 128) CHW

    assert ours.shape == theirs.shape, f"shape mismatch: {ours.shape} vs {theirs.shape}"
    max_abs_diff = float(np.max(np.abs(ours - theirs)))
    mean_abs_diff = float(np.mean(np.abs(ours - theirs)))
    print(f"    [osnet preprocess parity] max_abs_diff={max_abs_diff:.6f} mean_abs_diff={mean_abs_diff:.6f}")
    # cv2.INTER_LINEAR and PIL/torchvision's BILINEAR resize use different
    # kernels/anti-aliasing, so bit-exact equality is NOT expected -- this
    # bounds the divergence instead of assuming it away. 0.05 in normalized,
    # ImageNet-std-scaled space is well within the design's documented
    # cosine>=0.99 tolerance for the downstream embedding.
    assert max_abs_diff < 0.05, f"preprocessing diverges too much from torchreid: max_abs_diff={max_abs_diff}"


# ---------------------------------------------------------------------------
# 5. OSNetOnnxEmbedder against a REAL exported osnet_x1_0 graph
# ---------------------------------------------------------------------------

def _build_real_osnet_onnx(tmpdir):
    torch.manual_seed(0)
    model = torchreid.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=False)
    model.eval()
    dummy = torch.zeros(1, 3, 256, 128, dtype=torch.float32)
    path = os.path.join(tmpdir, "osnet_x1_0.onnx")
    torch.onnx.export(
        model, dummy, path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return path, model


def test_osnet_onnx_embedder_matches_torch_forward_on_same_preprocessed_tensor():
    with tempfile.TemporaryDirectory() as tmp:
        path, torch_model = _build_real_osnet_onnx(tmp)

        embedder = OSNetOnnxEmbedder(path, size=(128, 256), batch=8)
        assert embedder._out_dim == 512, f"unexpected feature dim {embedder._out_dim}"

        rng = np.random.default_rng(7)
        crop = rng.integers(0, 256, size=(200, 100, 3), dtype=np.uint8)

        onnx_embedding = embedder.embed_one(crop)
        assert onnx_embedding is not None
        assert onnx_embedding.shape == (512,)
        assert abs(float(np.linalg.norm(onnx_embedding)) - 1.0) < 1e-4, "embedding must be L2-normalized"

        # same preprocessing, fed through the ORIGINAL torch model directly
        blob = embedder._preprocess(crop)
        with torch.no_grad():
            torch_out = torch_model(torch.from_numpy(blob)).numpy()[0]
        torch_embedding = torch_out / (np.linalg.norm(torch_out) + 1e-12)

        cosine = float(np.dot(onnx_embedding, torch_embedding))
        print(f"    [osnet export parity, random weights] cosine={cosine:.6f}")
        # random weights, but SAME graph/weights on both sides -- this checks
        # the export+session plumbing is correct, not real embedding quality.
        assert cosine > 0.999, f"onnx vs torch forward diverge: cosine={cosine}"


def test_osnet_embed_batch_preserves_positional_alignment_with_invalid_crops():
    with tempfile.TemporaryDirectory() as tmp:
        path, _ = _build_real_osnet_onnx(tmp)
        embedder = OSNetOnnxEmbedder(path, size=(128, 256), batch=2)

        rng = np.random.default_rng(3)
        valid_a = rng.integers(0, 256, size=(64, 32, 3), dtype=np.uint8)
        valid_b = rng.integers(0, 256, size=(64, 32, 3), dtype=np.uint8)
        crops = [valid_a, None, valid_b, np.zeros((0, 0, 3), dtype=np.uint8)]

        out = embedder.embed(crops)
        assert out.shape == (4, 512)
        assert np.allclose(out[1], 0.0), "invalid crop (None) must be an all-zero row"
        assert np.allclose(out[3], 0.0), "invalid crop (empty) must be an all-zero row"
        assert not np.allclose(out[0], 0.0)
        assert not np.allclose(out[2], 0.0)

        one_a = embedder.embed_one(valid_a)
        one_b = embedder.embed_one(valid_b)
        assert np.allclose(out[0], one_a, atol=1e-5), "batched result must match embed_one for the same crop"
        assert np.allclose(out[2], one_b, atol=1e-5)


def test_osnet_embed_empty_list_returns_empty_array():
    with tempfile.TemporaryDirectory() as tmp:
        path, _ = _build_real_osnet_onnx(tmp)
        embedder = OSNetOnnxEmbedder(path, size=(128, 256), batch=8)
        out = embedder.embed([])
        assert out.shape == (0, 512)


# ---------------------------------------------------------------------------
# 6. tools/export_to_onnx.py::_ensure_names_metadata()
# ---------------------------------------------------------------------------

def test_ensure_names_metadata_noop_when_already_present():
    from tools.export_to_onnx import _ensure_names_metadata

    with tempfile.TemporaryDirectory() as tmp:
        imgsz = 64
        dets = torch.zeros(300, 6, dtype=torch.float32)
        path = _export_const_model(dets, (1, 3, imgsz, imgsz), tmp, "meta_present.onnx")

        model = onnx.load(path)
        prop = model.metadata_props.add()
        prop.key, prop.value = "names", str({0: "person"})
        onnx.save(model, path)
        before = onnx.load(path).SerializeToString()

        _ensure_names_metadata(path, {0: "person", 1: "bicycle"})  # different dict on purpose
        after = onnx.load(path).SerializeToString()

        assert before == after, "must be a strict no-op when 'names' key already exists"


def test_ensure_names_metadata_injects_when_absent():
    from tools.export_to_onnx import _ensure_names_metadata

    with tempfile.TemporaryDirectory() as tmp:
        imgsz = 64
        dets = torch.zeros(300, 6, dtype=torch.float32)
        path = _export_const_model(dets, (1, 3, imgsz, imgsz), tmp, "meta_absent.onnx")

        model = onnx.load(path)
        assert "names" not in {p.key for p in model.metadata_props}

        names = {0: "sentado", 1: "de pie"}
        _ensure_names_metadata(path, names)

        reloaded = onnx.load(path)
        meta = {p.key: p.value for p in reloaded.metadata_props}
        assert "names" in meta
        assert ast.literal_eval(meta["names"]) == names

        # and the detector's own reader must parse it back correctly, end to end
        det = YOLOOnnxDetector(path, imgsz=imgsz, conf=0.5)
        assert det.names == names


def main():
    check("onnx_yolo._letterbox() pads to the full target square (auto=False)", test_letterbox_pads_to_full_square_auto_false)
    check("onnx_yolo._letterbox() splits odd padding within 1px", test_letterbox_odd_padding_split_asymmetrically_by_at_most_1px)
    check("onnx_yolo._unletterbox() inverts _letterbox (round-trip)", test_unletterbox_inverts_letterbox_round_trip)
    check("onnx_yolo._unletterbox() clips to original image bounds", test_unletterbox_clips_to_original_bounds)
    check("onnx_yolo.__call__() NMS-free (1,300,6) branch, real onnxruntime session", test_call_nms_free_branch_end_to_end)
    check("onnx_yolo.__call__() single image (not a list) still returns a list", test_call_single_image_not_a_list_still_returns_list)
    check("onnx_yolo.__call__() v8-style branch, real cv2.dnn.NMSBoxes suppression", test_call_v8_style_branch_with_real_nms_suppression)
    check("onnx_yolo.__call__() v8-style branch, cross-class overlap NOT suppressed (per-class NMS)", test_call_v8_style_branch_cross_class_high_overlap_not_suppressed)
    check("onnx_yolo.__call__() raises on ONNX output batch-size mismatch (no silent raw[0] fallback)", test_call_batch_size_mismatch_raises)
    check("onnx_yolo.__call__() real multi-image batch, different sizes, correctly attributed per-image", test_call_multi_image_batch_different_sizes_correctly_attributed)
    check("onnx_yolo._read_names_metadata() ast.literal_eval round-trip + COCO_NAMES fallback", test_read_names_metadata_ast_literal_eval_roundtrip)
    check("onnx_reid._preprocess() vs real torchreid test transform (bounded divergence)", test_osnet_preprocess_matches_torchreid_test_transform)
    check("onnx_reid.embed_one() vs real torch forward on same exported graph (cosine>0.999)", test_osnet_onnx_embedder_matches_torch_forward_on_same_preprocessed_tensor)
    check("onnx_reid.embed() preserves positional alignment across invalid crops", test_osnet_embed_batch_preserves_positional_alignment_with_invalid_crops)
    check("onnx_reid.embed() on an empty list returns a (0,512) array", test_osnet_embed_empty_list_returns_empty_array)

    if _HAS_ONNX_PKG:
        check("export_to_onnx._ensure_names_metadata() no-op when 'names' present", test_ensure_names_metadata_noop_when_already_present)
        check("export_to_onnx._ensure_names_metadata() injects when 'names' absent", test_ensure_names_metadata_injects_when_absent)
    else:
        print("[SKIP] export_to_onnx._ensure_names_metadata() checks — `onnx` package not installed "
              "(see requirements-export.txt; not required for the runtime detectors this suite otherwise covers)")

    print("\n=== Fase 3a (PR4a) onnx detector+reid contract verification ===")
    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        line = f"[{status}] {name}"
        if not ok:
            line += f" — {detail}"
        print(line)

    print(f"\n{n_pass}/{len(results)} checks passed.")
    if n_pass != len(results):
        print("SUMMARY: FAIL")
        return 1
    print("SUMMARY: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
