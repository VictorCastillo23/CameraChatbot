"""Manual verification for the Fase 3a / PR4b contract (pose classifier ONNX
wrapper + cosine-distance DBSCAN replacement).

Not a pytest test: this repo has no test runner, so this is a plain
runnable script using bare ``assert`` statements, printing a PASS/FAIL
summary and exiting 0/1. Run it directly:

    python tests_manual/test_fase3a_pose_clustering_parity.py

This sandbox has real `torch`, `ultralytics`, `torchreid`, `onnxruntime`,
`torchvision`, `cv2`, and `scikit-learn` installed (confirmed at PR4a apply
time and re-confirmed here) — so, matching PR4a's verification bar, this
script goes well beyond `python -m py_compile`:

1. `PoseClsOnnxClassifier._preprocess()` measured, pixel-level, against the
   ACTUAL installed `ultralytics.data.augment.classify_transforms(size=224)`
   Compose object (real `torchvision.transforms.Resize` +
   `torchvision.transforms.CenterCrop`, not a hand-derived guess) on real
   synthetic images of several aspect ratios. This is the design's hazard
   #1 concern, verified for real: max/mean absolute pixel-value difference
   is measured and asserted below a tolerance (small, expected divergence
   from `cv2.INTER_LINEAR` vs. torchvision's antialiased bilinear kernel —
   the same category of measured-not-assumed divergence PR4a documented for
   `OSNetOnnxEmbedder._preprocess()` vs. real `torchreid` transforms), not
   silently assumed to match.
2. `PoseClsOnnxClassifier.predict()` end-to-end against a REAL exported
   ONNX classification model (`torch.onnx.export`, `dynamo=False`) whose
   `forward()` deliberately ends in `x.softmax(1)` — mirroring
   `ultralytics/nn/modules/head.py::Classify.forward()`'s confirmed
   export-mode behavior (`y = x.softmax(1); return y if self.export else
   (y, x)`) — compared against that SAME model's own direct `torch` forward
   pass on the identical preprocessed tensor. This is the hazard #4 check:
   if `predict()` applied softmax a second time, `conf` would not match the
   torch reference and this test would fail.
3. Batching: `predict()` called with more crops than `self.batch`, verifying
   chunked `onnxruntime` calls preserve positional order.
4. `_read_names_metadata()` against a real ONNX file with `metadata_props`
   written the same way `onnx_yolo.py`/`tools/export_to_onnx.py` do.
5. `cluster_cosine()` — the single most important correctness check in this
   PR — compared against REAL `sklearn.cluster.DBSCAN(eps, min_samples,
   metric="cosine").fit_predict(E)` across 20 random seeds and several
   `(eps, min_samples)` configurations, on synthetic L2-normalized
   embedding clusters (several tight groups of random unit vectors, plus
   scattered noise points, built with enough inter-cluster separation and
   enough intra-cluster tightness that no point sits exactly on a
   sklearn-documented border-point tie — see the note below on how ties are
   handled). Asserted via CO-MEMBERSHIP MATRIX equality (for every pair
   `(i, j)`: `mine[i] == mine[j] and mine[i] != -1` must equal the same
   predicate for sklearn's labels), which is invariant to arbitrary label
   *numbering* differences between the two implementations, plus an exact
   per-point noise-flag (`== -1`) match. This is stronger than eyeballing
   cluster counts: it fails on ANY point ending up in the wrong
   cluster/partition, not just a wrong total.

   Border-point tie-breaking note: sklearn's own docs state that when a
   border point is density-reachable from two mutually-disconnected
   clusters, which cluster claims it is scan-order-dependent (and this
   module's own docstring states the same, matching sklearn's own
   `dbscan_inner` flood-fill exactly). This test's synthetic data is
   deliberately constructed so that does not happen (well-separated
   cluster centers, tight per-cluster spread, `eps` chosen well below the
   inter-cluster distance) — so a co-membership mismatch here indicates a
   REAL clustering bug, not tie-breaking nondeterminism. Ambiguous
   border-tie scenarios are explicitly NOT constructed or asserted exact
   here, per the design's own documented acceptance of that nondeterminism.
6. `cluster_cosine()` empty-input and single-point edge cases.

Uses only synthetic/random-weight data throughout — never touches
`models/trained_yolo11m.pt` (not present in this sandbox) or the real
gallery files.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.cluster import DBSCAN  # noqa: E402

from camerachatbot.detectors.onnx_pose_classifier import PoseClsOnnxClassifier  # noqa: E402
from camerachatbot.identity.cosine_clustering import cluster_cosine  # noqa: E402

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
# Synthetic ONNX classifier model
# ---------------------------------------------------------------------------

class _TinyClsHead(torch.nn.Module):
    """Small conv+pool+linear+softmax network standing in for the real
    `trained_yolo11m.pt` classification head. The trailing `x.softmax(1)`
    matches `ultralytics/nn/modules/head.py::Classify.forward()`'s
    confirmed export-mode output (already-softmax'd probabilities) — this
    is what makes `predict()`'s "don't double-softmax" behavior testable.
    """

    def __init__(self, nc=3):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, stride=2, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.linear = torch.nn.Linear(4, nc)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        x = self.linear(x)
        return x.softmax(1)


def _export_tiny_cls_head(tmpdir, nc=3, seed=0):
    torch.manual_seed(seed)
    model = _TinyClsHead(nc=nc)
    model.eval()
    dummy = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    path = os.path.join(tmpdir, "tiny_cls.onnx")
    torch.onnx.export(
        model, dummy, path,
        input_names=["images"], output_names=["output"],
        dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        # See tools/export_to_onnx.py::export_reid() — recent torch defaults
        # to the dynamo exporter, which needs the separate `onnxscript`
        # package this project does not pin anywhere.
        dynamo=False,
    )
    return path, model


def _random_crops(rng, n=7, base_h=80, base_w=60):
    return [
        rng.integers(0, 256, size=(base_h + 10 * i, base_w + 5 * i, 3), dtype=np.uint8)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 1. Preprocessing parity against the REAL installed ultralytics transform
# ---------------------------------------------------------------------------

def test_preprocess_matches_real_classify_transforms():
    from ultralytics.data.augment import classify_transforms

    clf = PoseClsOnnxClassifier.__new__(PoseClsOnnxClassifier)  # no session needed for pure preprocessing
    clf.imgsz = 224

    ref_transform = classify_transforms(size=224)

    rng = np.random.default_rng(11)
    shapes = [(300, 150), (150, 300), (224, 224), (500, 500), (90, 400)]  # tall, wide, square, big square, very wide

    for h, w in shapes:
        crop_bgr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        # Real person crops have spatially-correlated pixel content
        # (photos), unlike literal i.i.d. random noise. A mild Gaussian
        # blur approximates that correlation cheaply — kept for realistic
        # test content even though `_preprocess()` now replicates
        # `PIL.Image.resize(size, Image.BILINEAR)` exactly (see below), so
        # the resize itself no longer diverges on any input, blurred or not.
        crop_bgr = cv2.GaussianBlur(crop_bgr, (9, 9), 3)
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        ref = ref_transform(Image.fromarray(rgb)).numpy()  # (3, 224, 224) float32 in [0, 1]
        mine = clf._preprocess([crop_bgr])[0]  # (3, 224, 224)

        assert ref.shape == mine.shape, f"shape mismatch for {(h, w)}: {ref.shape} vs {mine.shape}"

        diff = np.abs(ref - mine)
        # `_preprocess()` resizes via `PIL.Image.resize(size, Image.BILINEAR)`
        # directly, matching Ultralytics' real preprocessing exactly:
        # `ClassificationPredictor.preprocess()` wraps each crop in a
        # `PIL.Image` before `classify_transforms()` runs, and torchvision's
        # `Resize` dispatches PIL inputs to `img.resize(size, BILINEAR)` —
        # Pillow's own C resize, not torchvision's tensor-path kernel. This
        # produces a bit-for-bit 0.0 diff against the real reference
        # transform (confirmed on real keyFrames images via
        # `tools/verify_onnx_parity.py`: posecls prob_diff min=mean=max=0.0000
        # on all 14 images). The tolerance below is only for float32
        # numerical noise, not an approximation gap.
        assert diff.max() < 1e-5, f"max pixel diff too high for {(h, w)}: {diff.max()}"
        assert diff.mean() < 1e-5, f"mean pixel diff too high for {(h, w)}: {diff.mean()}"


# ---------------------------------------------------------------------------
# 2/3. predict() end-to-end: no double-softmax, correct batching/order
# ---------------------------------------------------------------------------

def test_predict_matches_torch_forward_no_double_softmax():
    with tempfile.TemporaryDirectory() as tmp:
        path, torch_model = _export_tiny_cls_head(tmp, nc=3, seed=0)

        clf = PoseClsOnnxClassifier(path, imgsz=224, batch=4, names={0: "sit", 1: "stand", 2: "other"})

        rng = np.random.default_rng(5)
        crops = _random_crops(rng, n=7)  # deliberately > batch=4, exercises chunking

        preds = clf.predict(crops)
        assert len(preds) == len(crops)

        blob = clf._preprocess(crops)
        with torch.no_grad():
            torch_out = torch_model(torch.from_numpy(blob)).numpy()  # already softmax'd, like the real export

        for i, (idx, conf) in enumerate(preds):
            exp_idx = int(np.argmax(torch_out[i]))
            exp_conf = float(torch_out[i, exp_idx])
            assert idx == exp_idx, f"crop {i}: pred_idx {idx} != torch argmax {exp_idx}"
            # If predict() applied softmax a SECOND time on top of the
            # already-softmax'd ONNX output, conf would be compressed toward
            # 1/nc and this would fail.
            assert abs(conf - exp_conf) < 1e-4, f"crop {i}: conf {conf} != torch conf {exp_conf}"

        # Confidences must be plausible probabilities (proof no re-softmax
        # collapsed them below any sane range) and each row sums close to 1.
        row_sums = torch_out.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-4), f"reference softmax rows do not sum to 1: {row_sums}"


def test_predict_empty_crops_returns_empty_list():
    with tempfile.TemporaryDirectory() as tmp:
        path, _ = _export_tiny_cls_head(tmp, nc=2, seed=1)
        clf = PoseClsOnnxClassifier(path, imgsz=224, batch=4)
        assert clf.predict([]) == []


# ---------------------------------------------------------------------------
# 4. names metadata reading
# ---------------------------------------------------------------------------

def test_read_names_metadata_roundtrip():
    if not _HAS_ONNX_PKG:
        raise AssertionError("onnx package not installed — cannot test metadata roundtrip")

    with tempfile.TemporaryDirectory() as tmp:
        path, _ = _export_tiny_cls_head(tmp, nc=2, seed=2)

        names = {0: "sentado", 1: "de pie"}
        model = onnx.load(path)
        prop = model.metadata_props.add()
        prop.key, prop.value = "names", str(names)
        onnx.save(model, path)

        clf = PoseClsOnnxClassifier(path, imgsz=224, batch=4)  # names=None -> read from metadata
        assert clf.names == names, f"expected {names}, got {clf.names}"


def test_explicit_names_override_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        path, _ = _export_tiny_cls_head(tmp, nc=2, seed=3)
        override = {0: "a", 1: "b"}
        clf = PoseClsOnnxClassifier(path, imgsz=224, batch=4, names=override)
        assert clf.names == override


# ---------------------------------------------------------------------------
# 5. cluster_cosine() vs REAL sklearn.cluster.DBSCAN
# ---------------------------------------------------------------------------

def _make_synthetic_embeddings(rng, n_clusters=3, per_cluster=8, dim=16, noise_pts=5, spread=0.05):
    """Several tight, well-separated unit-vector clusters + scattered noise.

    `spread` is small enough (and cluster centers random+high-dim enough)
    that intra-cluster cosine distances stay well below any reasonable
    `eps`, and inter-cluster/noise distances stay well above it — avoiding
    the border-point-tie ambiguity this test's docstring documents as
    out of scope.
    """
    vecs = []
    for _ in range(n_clusters):
        center = rng.normal(size=dim)
        center /= np.linalg.norm(center)
        for _ in range(per_cluster):
            v = center + rng.normal(scale=spread, size=dim)
            v /= np.linalg.norm(v)
            vecs.append(v)
    for _ in range(noise_pts):
        v = rng.normal(size=dim)
        v /= np.linalg.norm(v)
        vecs.append(v)
    return np.array(vecs, dtype=np.float32)


def _assert_same_partition(mine, theirs):
    n = len(mine)
    assert len(theirs) == n

    # Noise flags must match exactly, point for point.
    for i in range(n):
        assert (mine[i] == -1) == (theirs[i] == -1), (
            f"noise-flag mismatch at index {i}: mine={mine[i]}, sklearn={theirs[i]}"
        )

    # Co-membership matrix equality — invariant to arbitrary label renumbering.
    for i in range(n):
        for j in range(i + 1, n):
            a = bool(mine[i] == mine[j] and mine[i] != -1)
            b = bool(theirs[i] == theirs[j] and theirs[i] != -1)
            assert a == b, (
                f"co-membership mismatch for pair ({i}, {j}): "
                f"mine says {'same' if a else 'different'} cluster, "
                f"sklearn says {'same' if b else 'different'} cluster "
                f"(mine={mine.tolist()}, sklearn={theirs.tolist()})"
            )


def test_cluster_cosine_matches_sklearn_dbscan_across_seeds_and_configs():
    configs = [
        dict(eps=0.3, min_samples=4),
        dict(eps=0.5, min_samples=4),
        dict(eps=0.4, min_samples=3),
        dict(eps=0.2, min_samples=5),
    ]
    for seed in range(20):
        rng = np.random.default_rng(seed)
        E = _make_synthetic_embeddings(rng)
        for cfg in configs:
            mine = cluster_cosine(E, eps=cfg["eps"], min_samples=cfg["min_samples"])
            theirs = DBSCAN(eps=cfg["eps"], min_samples=cfg["min_samples"], metric="cosine").fit_predict(E)
            _assert_same_partition(mine, theirs)


def test_cluster_cosine_off_by_one_core_point_definition():
    """Regression guard for the design's own explicit off-by-one warning:
    core-point count must NOT subtract the point itself.

    Construct exactly `min_samples` points within `eps` of each other
    (including self) — sklearn counts self, so each of these points IS
    core, and they form exactly one cluster. If `cluster_cosine` wrongly
    subtracted 1 from the neighbor count, every point here would need
    `min_samples + 1` *other* neighbors to be core, making all of them
    noise instead (a different, wrong, result caught by this test).
    """
    dim = 8
    min_samples = 4
    rng = np.random.default_rng(42)
    center = rng.normal(size=dim)
    center /= np.linalg.norm(center)

    # exactly min_samples points, extremely tight (near-identical) so all
    # pairwise cosine distances are ~0, well under any eps >= 0.01.
    E = np.stack([center + rng.normal(scale=1e-4, size=dim) for _ in range(min_samples)])
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    E = E.astype(np.float32)

    labels = cluster_cosine(E, eps=0.01, min_samples=min_samples)
    assert (labels != -1).all(), f"expected all {min_samples} tight points to be clustered, got {labels}"
    assert len(set(labels.tolist())) == 1, f"expected exactly one cluster, got labels {labels}"

    sklearn_labels = DBSCAN(eps=0.01, min_samples=min_samples, metric="cosine").fit_predict(E)
    assert (sklearn_labels != -1).all(), "sanity check: sklearn itself should also cluster these as core"


def test_cluster_cosine_empty_and_single_point():
    empty = cluster_cosine(np.zeros((0, 8), dtype=np.float32))
    assert empty.shape == (0,)

    single = cluster_cosine(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), eps=0.5, min_samples=4)
    # A single point can never satisfy min_samples=4 (only itself as a
    # neighbor) -> must be noise.
    assert single.tolist() == [-1]


def main():
    check("PoseClsOnnxClassifier._preprocess() matches real classify_transforms(224)", test_preprocess_matches_real_classify_transforms)
    check("PoseClsOnnxClassifier.predict() matches torch forward, no double-softmax", test_predict_matches_torch_forward_no_double_softmax)
    check("PoseClsOnnxClassifier.predict() empty crops -> empty list", test_predict_empty_crops_returns_empty_list)
    check("PoseClsOnnxClassifier._read_names_metadata() roundtrip", test_read_names_metadata_roundtrip)
    check("PoseClsOnnxClassifier explicit names= overrides metadata", test_explicit_names_override_metadata)
    check("cluster_cosine() matches real sklearn.cluster.DBSCAN (20 seeds x 4 configs)", test_cluster_cosine_matches_sklearn_dbscan_across_seeds_and_configs)
    check("cluster_cosine() core-point off-by-one regression guard", test_cluster_cosine_off_by_one_core_point_definition)
    check("cluster_cosine() empty / single-point edge cases", test_cluster_cosine_empty_and_single_point)

    print("\n=== Fase 3a / PR4b (pose classifier + cosine clustering) verification ===")
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
