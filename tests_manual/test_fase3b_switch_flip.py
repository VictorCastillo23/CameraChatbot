"""Manual verification for the Fase 3b / PR5 switch-flip isinstance branching.

Not a pytest test: this repo has no test runner. Plain runnable script with
bare ``assert`` statements, printing a PASS/FAIL summary and exiting 0/1.

    python tests_manual/test_fase3b_switch_flip.py

PR5's gate review flagged a BLOCKER: PR5 flipped the default production
inference path (torch -> ONNX) in `bootstrap.py` and introduced isinstance-
based branching in three places to keep BOTH loader paths working (the
"real, tested rollback" the PR claims) — but shipped nothing checked into
the repo that automatically re-asserts either branch is actually reachable
and correctly selected. This file is that check, covering exactly the three
sites the reviewer named:

1. `camerachatbot/detectors/person_reid.py::YOLOPersonReID.__init__` /
   `.extract_embedding()` — `isinstance(reid_model, OSNetOnnxEmbedder)`.
2. `camerachatbot/detectors/pose_action_classifier.py::PoseActionClassifier
   .__init__` / `.run_on_json()` — `isinstance(self.model,
   PoseClsOnnxClassifier)`.
3. `camerachatbot/runtime/bootstrap.py::init_runtime()` — tuple-vs-bare-
   object shape absorption of `load_reid()`'s two differing return shapes
   (torch loader: `(reid_model, transform)` tuple; ONNX loader: bare
   `OSNetOnnxEmbedder`).

No real model weights or ONNX sessions are needed: ONNX-side stubs are real
`OSNetOnnxEmbedder`/`PoseClsOnnxClassifier` instances built via `__new__`
(skipping `__init__`, so no `onnxruntime.InferenceSession` is ever opened),
with only the attributes/methods the branch under test actually touches set
by hand on the instance. Because those attributes are set as plain instance
attributes (not class methods), calling them does not implicitly bind
`self` — the stub lambdas below intentionally do not take a leading `self`
argument. Torch-side stubs are genuine `torch.nn.Module` subclasses (real
`torch` is already an installed dependency in this repo, pending Fase 3c's
future removal) so `.eval()`/`.to()`/`__call__` are exercised for real, not
faked. `bootstrap.init_runtime()` itself is monkeypatched at its loader/
network seams (`load_detector`/`load_posecls`/`load_reid`/
`load_face_attr_sessions`/`build_supabase`) so it runs to completion
in-process without touching `.env`, real model files, or a network
connection — every patched attribute is restored in a `finally` block
regardless of outcome.

Each test below is written to genuinely fail if the wrong branch runs: the
ONNX-shaped stubs deliberately lack the methods the legacy torch branch
calls (`.to()`, `.fuse()`, `__call__`), so a misrouted branch raises
`AttributeError`/`TypeError` rather than silently passing.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from camerachatbot.detectors.onnx_reid import OSNetOnnxEmbedder  # noqa: E402
from camerachatbot.detectors.onnx_pose_classifier import PoseClsOnnxClassifier  # noqa: E402
from camerachatbot.detectors.person_reid import YOLOPersonReID  # noqa: E402
from camerachatbot.detectors.pose_action_classifier import PoseActionClassifier  # noqa: E402
from camerachatbot.runtime import bootstrap  # noqa: E402

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
# Stubs
# ---------------------------------------------------------------------------

def _onnx_reid_stub(vector):
    """A real `OSNetOnnxEmbedder` instance (isinstance-true) with no ONNX
    session — `__new__` skips `__init__`. `embed_one` is set directly on the
    instance as a fixed-vector stub; `OSNetOnnxEmbedder` defines neither
    `.eval()` nor `.to()`, so routing this into the legacy torch branch
    would raise `AttributeError` immediately.
    """
    stub = OSNetOnnxEmbedder.__new__(OSNetOnnxEmbedder)
    stub.embed_one = lambda crop: np.asarray(vector, dtype=np.float32)
    return stub


class _FakeTorchReidModel(torch.nn.Module):
    """Real `torch.nn.Module` — has `.eval()`/`.to()` for real, and is NOT
    an `OSNetOnnxEmbedder` instance, so it must select the legacy branch.
    `forward()` ignores its input and returns a fixed unnormalized vector
    per batch item, so the expected L2-normalized output is known exactly.
    """

    def forward(self, x):
        b = x.shape[0]
        return torch.full((b, 8), 2.0)


def _onnx_pose_stub(names, predict_fn):
    """A real `PoseClsOnnxClassifier` instance (isinstance-true) with no
    ONNX session. `PoseClsOnnxClassifier` defines neither `.to()`, `.fuse()`
    nor `__call__`, so routing this into the legacy branch raises
    `AttributeError`/`TypeError` immediately.
    """
    stub = PoseClsOnnxClassifier.__new__(PoseClsOnnxClassifier)
    stub.names = names
    stub.predict = predict_fn
    return stub


class _FakeProbs:
    """Mimics ultralytics `Results.probs`: `.data` is whatever
    `.detach().cpu().numpy()` chains off of — a real torch tensor here, so
    that chain is exercised for real rather than mocked.
    """

    def __init__(self, scores):
        self.data = torch.tensor(scores, dtype=torch.float32)


class _FakeClsResult:
    def __init__(self, scores):
        self.probs = _FakeProbs(scores)


class _FakeYoloClsModel:
    """Legacy ultralytics-YOLO-CLS-shaped stub: NOT a `PoseClsOnnxClassifier`
    instance, has `.to()`/`.fuse()`/`.names`, and is callable with the exact
    kwargs `run_on_json()`'s legacy branch passes.
    """

    def __init__(self, scores_per_crop):
        self.names = {0: "sit", 1: "stand"}
        self._scores_per_crop = scores_per_crop
        self.to_called_with = "unset"
        self.fuse_called = False

    def to(self, device):
        self.to_called_with = device
        return self

    def fuse(self):
        self.fuse_called = True

    def __call__(self, crops, imgsz=None, batch=None, half=None, verbose=None):
        return [_FakeClsResult(self._scores_per_crop) for _ in crops]


def _write_frame_and_json(tmpdir, frame_id="frame1"):
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    frame_path = os.path.join(tmpdir, f"{frame_id}.jpg")
    cv2.imwrite(frame_path, img)

    # bbox area 2500 / frame area 4096 = ~0.61 -> comfortably above the
    # default min_area_ratio=0.04 gate, so this crop is never skipped.
    data = {frame_id: [{"kind": "person", "bbox": [0, 0, 50, 50], "attributes": {}}]}
    json_path = os.path.join(tmpdir, "tracking_results_1.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return json_path, frame_id


# ---------------------------------------------------------------------------
# 1. person_reid.py::YOLOPersonReID — isinstance(reid_model, OSNetOnnxEmbedder)
# ---------------------------------------------------------------------------

def test_person_reid_onnx_branch_skips_torch_lifecycle():
    with tempfile.TemporaryDirectory() as tmp:
        onnx_stub = _onnx_reid_stub([1.0, 0.0, 0.0])
        reid = YOLOPersonReID(model=None, output_folder=tmp, reid_model=onnx_stub, device=None)
        assert reid.reid_model is onnx_stub, "ONNX reid_model must be stored as-is, unmodified"
        assert reid.device is None, "device must NOT be resolved on the ONNX branch (no torch placement needed)"


def test_person_reid_onnx_branch_extract_embedding_uses_embed_one():
    with tempfile.TemporaryDirectory() as tmp:
        onnx_stub = _onnx_reid_stub([3.0, 4.0])  # deliberately NOT pre-normalized
        reid = YOLOPersonReID(model=None, output_folder=tmp, reid_model=onnx_stub, device=None)
        crop = np.zeros((20, 20, 3), dtype=np.uint8)
        out = reid.extract_embedding(crop)
        # extract_embedding must return embed_one()'s value verbatim on the
        # ONNX branch (no re-normalization, that's embed_one's own job) — if
        # the legacy branch ran instead this would crash on
        # `self.reid_transform is None` and return None.
        assert np.allclose(out, [3.0, 4.0]), f"expected embed_one()'s raw output, got {out}"


def test_person_reid_torch_branch_resolves_device_and_lifecycle():
    with tempfile.TemporaryDirectory() as tmp:
        torch_stub = _FakeTorchReidModel()
        reid = YOLOPersonReID(model=None, output_folder=tmp, transform=lambda img: torch.zeros(3, 4, 4),
                               reid_model=torch_stub, device=None)
        # Legacy branch must resolve a real device (None -> cpu/cuda) and
        # call `.eval().to(device)` — confirmed by the model no longer being
        # in training mode and by `self.device` no longer being None.
        assert reid.device is not None, "legacy branch must resolve device, not leave it None"
        assert isinstance(reid.device, torch.device)
        assert reid.reid_model.training is False, "legacy branch must call .eval()"


def test_person_reid_torch_branch_extract_embedding_uses_transform_and_model_call():
    with tempfile.TemporaryDirectory() as tmp:
        torch_stub = _FakeTorchReidModel()
        reid = YOLOPersonReID(model=None, output_folder=tmp,
                               transform=lambda img: torch.zeros(3, 4, 4),
                               reid_model=torch_stub, device=None)
        crop = np.zeros((10, 10, 3), dtype=np.uint8)
        out = reid.extract_embedding(crop)
        assert out is not None
        # _FakeTorchReidModel.forward returns a constant 8-dim [2,2,...,2]
        # vector; extract_embedding's legacy branch L2-normalizes it.
        expected = np.full(8, 2.0, dtype=np.float32)
        expected = expected / np.linalg.norm(expected)
        assert np.allclose(out, expected, atol=1e-5), f"expected normalized constant vector, got {out}"
        assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-5, "legacy branch must L2-normalize the raw model output"


def test_person_reid_torch_branch_with_no_transform_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        torch_stub = _FakeTorchReidModel()
        reid = YOLOPersonReID(model=None, output_folder=tmp, transform=None,
                               reid_model=torch_stub, device=None)
        crop = np.zeros((10, 10, 3), dtype=np.uint8)
        out = reid.extract_embedding(crop)
        # Proves this genuinely took the legacy branch (which depends on
        # reid_transform) rather than the ONNX branch (which ignores it
        # entirely and would have returned a real embedding here).
        assert out is None, "legacy branch with no transform must return None, not silently succeed"


# ---------------------------------------------------------------------------
# 2. pose_action_classifier.py::PoseActionClassifier — isinstance(self.model, PoseClsOnnxClassifier)
# ---------------------------------------------------------------------------

def test_pose_onnx_branch_skips_to_and_fuse():
    onnx_stub = _onnx_pose_stub({0: "sit", 1: "stand"}, predict_fn=lambda crops: [(1, 0.95)] * len(crops))
    clf = PoseActionClassifier(yolo=onnx_stub, device="cpu")
    assert clf.model is onnx_stub
    assert clf.names == {0: "sit", 1: "stand"}


def test_pose_legacy_branch_calls_to_and_fuse():
    legacy_stub = _FakeYoloClsModel(scores_per_crop=[0.05, 0.95])
    clf = PoseActionClassifier(yolo=legacy_stub, device="cpu")
    assert clf.model is legacy_stub
    assert legacy_stub.to_called_with == "cpu", "legacy branch must call .to(device)"
    assert legacy_stub.fuse_called is True, "legacy branch must call .fuse()"


def test_pose_run_on_json_onnx_branch_uses_predict():
    with tempfile.TemporaryDirectory() as tmp:
        json_path, frame_id = _write_frame_and_json(tmp)
        onnx_stub = _onnx_pose_stub({0: "sit", 1: "stand"},
                                     predict_fn=lambda crops: [(1, 0.95) for _ in crops])
        clf = PoseActionClassifier(yolo=onnx_stub, frames_folder=tmp, conf_threshold=0.6)
        # If run_on_json's isinstance check routed this into the legacy
        # branch, `self.model(crops, imgsz=..., ...)` would raise TypeError
        # — PoseClsOnnxClassifier has no `__call__`.
        out_file = clf.run_on_json(json_path)
        with open(out_file, "r", encoding="utf-8") as f:
            out = json.load(f)
        attrs = out[frame_id][0]["attributes"]
        assert attrs["pose"] == "de pie", f"expected mapped 'stand'->'de pie', got {attrs}"
        assert abs(attrs["pose_conf"] - 0.95) < 1e-6
        assert attrs["pose_source"] == "yolo-cls"


def test_pose_run_on_json_legacy_branch_uses_call_and_probs():
    with tempfile.TemporaryDirectory() as tmp:
        json_path, frame_id = _write_frame_and_json(tmp)
        legacy_stub = _FakeYoloClsModel(scores_per_crop=[0.05, 0.95])
        clf = PoseActionClassifier(yolo=legacy_stub, frames_folder=tmp, conf_threshold=0.6)
        # If run_on_json's isinstance check routed this into the ONNX
        # branch, `self.model.predict(crops)` would raise AttributeError —
        # _FakeYoloClsModel has no `.predict()`.
        out_file = clf.run_on_json(json_path)
        with open(out_file, "r", encoding="utf-8") as f:
            out = json.load(f)
        attrs = out[frame_id][0]["attributes"]
        assert attrs["pose"] == "de pie", f"expected argmax=1 ('stand')->'de pie' via .probs parsing, got {attrs}"
        assert abs(attrs["pose_conf"] - 0.95) < 1e-6
        assert attrs["pose_source"] == "yolo-cls"


# ---------------------------------------------------------------------------
# 3. bootstrap.py::init_runtime() — load_reid() tuple-vs-bare-object absorption
# ---------------------------------------------------------------------------

def _run_init_runtime_with_reid(reid_return_value):
    """Runs the REAL `init_runtime()` with its loader/network seams
    monkeypatched to hermetic stubs, restoring every patched attribute in a
    `finally` block regardless of outcome. This exercises the actual
    `isinstance(reid_loaded, tuple)` absorption logic inside `init_runtime()`
    — not a reimplementation of it.
    """
    originals = {
        name: getattr(bootstrap, name)
        for name in ("build_supabase", "load_detector", "load_posecls", "load_reid", "load_face_attr_sessions")
    }
    try:
        bootstrap.build_supabase = lambda: (None, None)
        bootstrap.load_detector = lambda: "fake_detector"
        bootstrap.load_posecls = lambda: "fake_posecls"
        bootstrap.load_reid = lambda: reid_return_value
        bootstrap.load_face_attr_sessions = lambda: ((None, None, None), (None, None, None))
        return bootstrap.init_runtime()
    finally:
        for name, fn in originals.items():
            setattr(bootstrap, name, fn)


def test_bootstrap_absorbs_tuple_shape_from_torch_loader():
    fake_transform = object()
    runtime = _run_init_runtime_with_reid(("fake_torch_reid_model", fake_transform))
    assert runtime["reid_model"] == "fake_torch_reid_model"
    assert runtime["reid_transform"] is fake_transform, "tuple shape must be unpacked into (reid_model, reid_transform)"
    assert "device" not in runtime, "RUNTIME['device'] was removed in Fase 3b — onnx_providers() is now the sole device-selection path"


def test_bootstrap_absorbs_bare_object_shape_from_onnx_loader():
    onnx_stub = _onnx_reid_stub([1.0])
    runtime = _run_init_runtime_with_reid(onnx_stub)
    assert runtime["reid_model"] is onnx_stub
    assert "reid_transform" in runtime, "reid_transform key must still be present (not removed outright) even on the bare-object/ONNX shape"
    assert runtime["reid_transform"] is None, "bare-object shape must resolve reid_transform to None, not crash or omit the key"
    assert "device" not in runtime


def test_bootstrap_seams_restored_after_each_run():
    # Sanity check on the test harness itself: confirm the monkeypatches
    # above don't leak into module state for any other consumer of
    # bootstrap.py (e.g. run_local.py / run_webhook.py importing it later).
    assert bootstrap.load_reid.__module__ == "camerachatbot.runtime.loaders_onnx", (
        "bootstrap.load_reid must be restored to the real loaders_onnx.load_reid "
        "after this suite's monkeypatching, not left pointing at a test stub"
    )


def main():
    check("person_reid.YOLOPersonReID.__init__ ONNX branch skips torch .eval()/.to() lifecycle", test_person_reid_onnx_branch_skips_torch_lifecycle)
    check("person_reid.extract_embedding() ONNX branch uses embed_one() verbatim", test_person_reid_onnx_branch_extract_embedding_uses_embed_one)
    check("person_reid.YOLOPersonReID.__init__ legacy branch resolves device + calls .eval()", test_person_reid_torch_branch_resolves_device_and_lifecycle)
    check("person_reid.extract_embedding() legacy branch uses reid_transform + model call + L2-normalizes", test_person_reid_torch_branch_extract_embedding_uses_transform_and_model_call)
    check("person_reid.extract_embedding() legacy branch with no transform returns None (routing proof)", test_person_reid_torch_branch_with_no_transform_returns_none)
    check("pose_action_classifier.PoseActionClassifier.__init__ ONNX branch skips .to()/.fuse()", test_pose_onnx_branch_skips_to_and_fuse)
    check("pose_action_classifier.PoseActionClassifier.__init__ legacy branch calls .to()/.fuse()", test_pose_legacy_branch_calls_to_and_fuse)
    check("pose_action_classifier.run_on_json() ONNX branch uses .predict()", test_pose_run_on_json_onnx_branch_uses_predict)
    check("pose_action_classifier.run_on_json() legacy branch uses __call__()+.probs parsing", test_pose_run_on_json_legacy_branch_uses_call_and_probs)
    check("bootstrap.init_runtime() absorbs tuple shape (loaders_torch rollback)", test_bootstrap_absorbs_tuple_shape_from_torch_loader)
    check("bootstrap.init_runtime() absorbs bare-object shape (loaders_onnx default)", test_bootstrap_absorbs_bare_object_shape_from_onnx_loader)
    check("bootstrap.py monkeypatched seams restored after use", test_bootstrap_seams_restored_after_each_run)

    print("\n=== Fase 3b (PR5) switch-flip isinstance branching verification ===")
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
