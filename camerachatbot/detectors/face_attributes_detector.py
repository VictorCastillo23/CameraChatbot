import os, json, cv2, numpy as np
from typing import Dict, List, Any, Optional, Tuple

# ---------- Utils básicos ----------
def _infer_layout_and_size(input_shape) -> Tuple[str, int, int, int]:
    if len(input_shape) != 4:
        raise ValueError(f"Esperado tensor 4D, recibido: {input_shape}")
    _, d1, d2, d3 = input_shape
    if (isinstance(d1, int) and d1 in (1, 3)) or (d1 is None):
        return "NCHW", int(d1 if isinstance(d1, int) else 3), int(d2 or 224), int(d3 or 224)
    if (isinstance(d3, int) and d3 in (1, 3)) or (d3 is None):
        return "NHWC", int(d3 if isinstance(d3, int) else 3), int(d1 or 224), int(d2 or 224)
    return "NCHW", int(d1 if isinstance(d1, int) else 3), int(d2 or 224), int(d3 or 224)

_DTYPE_MAP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(uint8)": np.uint8,
    "tensor(int8)": np.int8,
}

def _onnx_input_spec(sess) -> Tuple[str, int, int, int, np.dtype]:
    inp = sess.get_inputs()[0]
    layout, C, H, W = _infer_layout_and_size(inp.shape)
    dtype = _DTYPE_MAP.get(getattr(inp, "type", "tensor(float)"), np.float32)
    return layout, C, H, W, dtype

def _choose_interpolation(src_h, src_w, dst_h, dst_w):
    return cv2.INTER_CUBIC if (dst_h > src_h or dst_w > src_w) else cv2.INTER_AREA

def _expand_bbox(b, scale: float, img_w: int, img_h: int):
    x1, y1, x2, y2 = map(int, b)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1), (y2 - y1)
    w2, h2 = (w * scale) / 2.0, (h * scale) / 2.0
    nx1 = int(max(0, np.floor(cx - w2))); ny1 = int(max(0, np.floor(cy - h2)))
    nx2 = int(min(img_w, np.ceil(cx + w2))); ny2 = int(min(img_h, np.ceil(cy + h2)))
    return [nx1, ny1, nx2, ny2]

def _crop_face_from_img(img_bgr: np.ndarray, face_bbox, expand: float, min_face_size: int):
    h, w = img_bgr.shape[:2]
    if not isinstance(face_bbox, (list, tuple)) or len(face_bbox) != 4:
        return None, None, True
    ex = _expand_bbox(face_bbox, expand, w, h)
    x1, y1, x2, y2 = ex
    if x2 <= x1 or y2 <= y1:
        return None, ex, True
    fw, fh = (x2 - x1), (y2 - y1)
    lowres = (fw < min_face_size) or (fh < min_face_size)
    crop = img_bgr[y1:y2, x1:x2]
    return crop, ex, lowres

def _preprocess_bgr(
    src_bgr: np.ndarray,
    *, layout: str, C: int, H: int, W: int, dtype: np.dtype,
    expect_color: str = "RGB",
    mean=None, std=None,
    use_clahe: bool = True
) -> np.ndarray:
    if C == 1:
        gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
        interp = _choose_interpolation(*gray.shape, H, W)
        img = cv2.resize(gray, (W, H), interpolation=interp)
        if use_clahe:
            img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
        if np.issubdtype(dtype, np.floating):
            img = img.astype(np.float32) / 255.0
            m = float(mean[0]) if mean is not None else 0.5
            s = float(std[0]) if std is not None else 0.5
            if s == 0: s = 1.0
            img = (img - m) / s
        else:
            img = img.astype(dtype)
        blob = img[None, None, :, :] if layout == "NCHW" else img[:, :, None][None, ...]
        return blob.astype(np.float32) if dtype == np.float32 else blob

    # C == 3
    interp = _choose_interpolation(*src_bgr.shape[:2], H, W)
    img = cv2.resize(src_bgr, (W, H), interpolation=interp)
    if expect_color.upper() == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if np.issubdtype(dtype, np.floating):
        img = img.astype(np.float32) / 255.0
        if mean is not None and std is not None:
            mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
            std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
            std_arr[std_arr == 0] = 1.0
            img = (img - mean_arr) / std_arr
    else:
        img = img.astype(dtype)
    if layout == "NCHW":
        img = np.transpose(img, (2, 0, 1))
    blob = img[None, ...]
    return blob.astype(np.float32) if dtype == np.float32 else blob

# ---------- Detector ----------
class FaceAttributesDetector:
    def __init__(
        self,
        emotion_input: str, emotion_output: str,
        age_input: str, age_output: str,
        frames_folder: Optional[str] = None,
        emotion_labels: Optional[List[str]] = None,
        emotion_mean=None, emotion_std=None,
        age_mean=(0.485, 0.456, 0.406),
        age_std=(0.229, 0.224, 0.225),
        emotion_sess=None, age_sess=None,
        emotion_colorspace: str = "RGB",
        age_colorspace: str = "RGB",
        face_expand: float = 1.8,
        min_face_size: int = 48,
    ):
        self.frames_folder = frames_folder
        self.emotion_input, self.emotion_output = emotion_input, emotion_output
        self.age_input, self.age_output = age_input, age_output
        self.emotion_sess, self.age_sess = emotion_sess, age_sess
        self.emotion_labels = emotion_labels or ["neutral","feliz","triste","sorpresa","miedo","enojo","disgusto"]
        self.emotion_mean, self.emotion_std = emotion_mean, emotion_std
        self.age_mean, self.age_std = age_mean, age_std
        self.emotion_colorspace, self.age_colorspace = emotion_colorspace, age_colorspace
        self.face_expand, self.min_face_size = float(face_expand), int(min_face_size)

    def predict_emotion(self, face_bgr: np.ndarray) -> Dict[str, Any]:
        if self.emotion_sess is None:
            return {"label": None, "confidence": 0.0}
        layout, C, H, W, dtype = _onnx_input_spec(self.emotion_sess)
        blob = _preprocess_bgr(
            face_bgr, layout=layout, C=C, H=H, W=W, dtype=dtype,
            expect_color=self.emotion_colorspace,
            mean=self.emotion_mean, std=self.emotion_std,
            use_clahe=True
        )
        logits = self.emotion_sess.run([self.emotion_output], {self.emotion_input: blob})[0]
        logits = logits[0] if getattr(logits, "ndim", 1) > 1 else logits
        logits = logits - np.max(logits)
        p = np.exp(logits); probs = p / (np.sum(p) + 1e-8)
        idx = int(np.argmax(probs))
        label = self.emotion_labels[idx] if idx < len(self.emotion_labels) else f"class_{idx}"
        return {"label": label, "confidence": float(probs[idx])}

    @staticmethod
    def _softmax(x):
        x = x - np.max(x); e = np.exp(x); return e / (e.sum() + 1e-8)

    def predict_age(self, face_bgr: np.ndarray) -> Dict[str, Any]:
        if self.age_sess is None:
            return {"age": None, "adult": None, "confidence": 0.0}

        layout, C, H, W, dtype = _onnx_input_spec(self.age_sess)
        blob = _preprocess_bgr(
            face_bgr, layout=layout, C=C, H=H, W=W, dtype=dtype,
            expect_color=self.age_colorspace, mean=self.age_mean, std=self.age_std,
            use_clahe=False
        )
        out = self.age_sess.run([self.age_output], {self.age_input: blob})[0]
        out = np.array(out);
        out = out[0] if out.ndim > 1 else out

        # 3 rutas + confianza
        if out.ndim == 0 or out.size == 1:
            age = int(round(float(out.reshape(-1)[0])))
            conf = 1.0  # regression: sin prob explícita
        else:
            K = out.shape[-1]
            if K in (100, 101):
                p = self._softmax(out)
                ages = np.arange(K, dtype=np.float32)
                age = int(round(float((p * ages).sum())))
                conf = float(p.max())
            elif K in (8, 10):
                centers = np.array([1, 5, 10, 18, 28, 40.5, 50.5, 80.0], dtype=np.float32)[:K]
                p = self._softmax(out)
                age = int(round(float((p * centers).sum())))
                conf = float(p.max())
            else:
                p = self._softmax(out)
                age = int(np.argmax(p))
                conf = float(p.max())

        return {"age": age, "adult": bool(age >= 18), "confidence": conf}

    def run_on_json(self, json_file: str) -> str:
        with open(json_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        updated: Dict[str, List[Dict[str, Any]]] = {}
        for frame_id, entries in results.items():
            if "neighborhood" in frame_id:
                continue
            frame_path = os.path.join(self.frames_folder, f"{frame_id}.jpg")
            img = cv2.imread(frame_path)
            if img is None:
                updated[frame_id] = entries
                continue

            out_entries: List[Dict[str, Any]] = []
            for idx, e in enumerate(entries):
                if e.get("kind") != "person":
                    out_entries.append(e)
                    continue

                e.setdefault("attributes", {})
                face = e["attributes"].setdefault("face", {})
                bbox = face.get("bbox")

                # si no hay bbox → no borrar persona; deja None+conf=0
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    face.setdefault("emotion", {"label": None, "confidence": 0.0})
                    face.setdefault("age", None)
                    face.setdefault("adult", None)
                    face["age_confidence"] = 0.0
                    out_entries.append(e)
                    continue

                # 1) expandimos y recortamos
                crop, ex_bbox, lowres = _crop_face_from_img(
                    img, bbox, expand=self.face_expand, min_face_size=self.min_face_size
                )
                face["bbox_expanded"] = ex_bbox
                face["lowres"] = bool(lowres)

                # 2) si el expandido falló, intentar con el bbox original sin expandir (fallback)
                if crop is None or crop.size == 0:
                    x1, y1, x2, y2 = map(int, bbox)
                    h, w = img.shape[:2]
                    x1 = max(0, min(x1, w - 1));
                    x2 = max(0, min(x2, w))
                    y1 = max(0, min(y1, h - 1));
                    y2 = max(0, min(y2, h))
                    if x2 > x1 and y2 > y1:
                        crop = img[y1:y2, x1:x2]

                # 3) si AÚN no hay crop viable → deja None + conf=0
                if crop is None or crop.size == 0:
                    face["emotion"] = {"label": None, "confidence": 0.0}
                    face["age"] = None
                    face["adult"] = None
                    face["age_confidence"] = 0.0
                    out_entries.append(e)
                    continue

                # 4) SIEMPRE predecimos, incluso si lowres=True
                emo = self.predict_emotion(crop)  # {"label","confidence"}
                age_info = self.predict_age(crop)  # {"age","adult","confidence"}

                face["emotion"] = emo
                face["age"] = age_info["age"]
                face["adult"] = age_info["adult"]
                face["age_confidence"] = age_info.get("confidence", 0.0)

                out_entries.append(e)

            updated[frame_id] = out_entries

        updated['neighborhood'] = results.setdefault("neighborhood", {})

        out_path = os.path.join(os.path.dirname(json_file), "tracking_face_5.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(updated, f, indent=4, ensure_ascii=False)
        return out_path
