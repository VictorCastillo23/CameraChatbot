"""ONNX Runtime replacement for the torchreid OSNet Re-ID embedder.

`OSNetOnnxEmbedder` reproduces the exact preprocessing
`torchreid.data.transforms.build_transforms(height=256, width=128)`'s TEST
path applies (confirmed by reading the installed `torchreid==0.2.5` source,
`torchreid/reid/data/transforms.py::build_transforms` — this is not a
generic-ImageNet guess):

    test transform = Compose([Resize((256, 128)), ToTensor(), Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

and `person_reid.py::extract_embedding()`'s own preprocessing before that
transform: `Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))` — i.e.
BGR->RGB happens BEFORE resize/normalize. Any mismatch here surfaces as
cosine similarity < 0.99 in `tools/verify_onnx_parity.py`'s ReID check.

Inert module: `YOLOPersonReID.extract_embedding()` still uses
`torchreid`/`self.reid_model` directly (see `camerachatbot/detectors/
person_reid.py`) — nothing wires this in until Fase 3b's `embed_one()`
adapter swap, gated on the user's parity sign-off.
"""

import cv2
import numpy as np
import onnxruntime as ort

# ImageNet mean/std, exactly as `torchreid.data.transforms.build_transforms`'s
# defaults (`norm_mean`/`norm_std` kwargs) — NOT re-derived, read verbatim
# from the installed torchreid source to avoid a silent constant mismatch.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_DEFAULT_OUT_DIM = 512  # osnet_x1_0 feature dim; overridden from the ONNX output shape when static


class OSNetOnnxEmbedder:
    """ONNX Runtime OSNet Re-ID embedder.

    `size` is `(width, height)` — matches `cv2.resize(img, size)`'s
    argument order, and is `(128, 256)` by default to mirror torchreid's
    `build_transforms(height=256, width=128)` call in `bootstrap.py::
    load_reid()`.
    """

    def __init__(self, onnx_path, providers=None, size=(128, 256), batch=32):
        self.session = ort.InferenceSession(
            str(onnx_path), providers=providers or ["CPUExecutionProvider"]
        )
        self.size = tuple(size)  # (W, H)
        self.batch = int(batch)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        out_shape = self.session.get_outputs()[0].shape
        last_dim = out_shape[-1] if out_shape else None
        self._out_dim = int(last_dim) if isinstance(last_dim, int) else _DEFAULT_OUT_DIM

    def _preprocess(self, crop_bgr):
        """BGR -> RGB -> resize (bilinear) -> /255 -> ImageNet normalize -> NCHW."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.size, interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        return arr[None, ...]  # NCHW, N=1

    @staticmethod
    def _l2_normalize(x, axis=-1, eps=1e-12):
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        return (x / (norm + eps)).astype(np.float32)

    @staticmethod
    def is_valid_embedding(vec: np.ndarray, eps: float = 1e-9) -> bool:
        """`True` unless `vec` is the all-zero sentinel `embed()`/`embed_one()`
        use for an invalid input crop (norm 0, NOT unit-norm — see those
        methods' docstrings). Cheap helper for callers doing cosine
        similarity: `np.dot(a, b)` against a zero-norm row is always `0.0`,
        which silently looks like "no match" rather than "no data" unless
        checked explicitly.
        """
        return bool(np.linalg.norm(vec) > eps)

    def embed_one(self, crop: np.ndarray):
        """Embed a single BGR crop. Returns `(512,)` float32, L2-normalized, or `None`.

        `None` is the invalid-crop signal here (caller must check for it
        before doing anything downstream, e.g. cosine similarity).
        """
        if crop is None or crop.size == 0:
            return None
        blob = self._preprocess(crop)
        out = self.session.run([self.output_name], {self.input_name: blob})[0]
        feat = np.asarray(out[0], dtype=np.float32)
        return self._l2_normalize(feat)

    def embed(self, crops: list) -> np.ndarray:
        """Embed a list of BGR crops in mini-batches of `self.batch`.

        Returns `(len(crops), 512)` float32. Rows are L2-normalized (unit
        norm) EXCEPT rows corresponding to an invalid input crop (`None` or
        zero-size), which are a literal all-zero vector (norm 0, not
        unit-norm) — invalid entries are skipped from ONNX inference but
        still occupy their row, so callers that assume `embeddings[i]`
        corresponds to `crops[i]` (positional correspondence) are never
        silently desynced by a dropped crop. This mirrors `embed_one()`'s
        `None` signal for a single invalid crop, without collapsing the
        batch shape.

        Callers doing cosine similarity (`np.dot`) MUST treat zero-norm rows
        as "no match" rather than compute a cosine against them — a raw dot
        product against an all-zero row is always `0.0`, which looks
        identical to "compared but dissimilar" unless checked. Use
        `OSNetOnnxEmbedder.is_valid_embedding(row)` (or `np.linalg.norm(row)
        > 0`) to distinguish the two cases.
        """
        n = len(crops)
        embeddings = np.zeros((n, self._out_dim), dtype=np.float32)
        if n == 0:
            return embeddings

        valid_idx, blobs = [], []
        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                continue
            blobs.append(self._preprocess(crop))
            valid_idx.append(i)

        for start in range(0, len(blobs), self.batch):
            chunk_blobs = blobs[start:start + self.batch]
            chunk_idx = valid_idx[start:start + self.batch]

            batch_arr = np.concatenate(chunk_blobs, axis=0)
            out = self.session.run([self.output_name], {self.input_name: batch_arr})[0]
            out = self._l2_normalize(np.asarray(out, dtype=np.float32), axis=1)

            for j, idx in enumerate(chunk_idx):
                embeddings[idx] = out[j]

        return embeddings
