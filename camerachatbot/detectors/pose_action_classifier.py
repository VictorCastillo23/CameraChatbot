import os, cv2, json

from camerachatbot.detectors.onnx_pose_classifier import PoseClsOnnxClassifier

class PoseActionClassifier():
    def __init__(self, yolo, frames_folder=None, conf_threshold=0.60, class_map=None,
                 imgsz=224, batch=16, res=None):
        self.model = yolo
        self.frames_folder = frames_folder
        self.confs = conf_threshold
        self.imgsz = imgsz
        self.batch = batch
        self.res = res  # opcional; si no, llama set_res() luego
        self._frames_map = None

        # Fase 3c: `yolo` is always a `PoseClsOnnxClassifier` — the only
        # loader module left is `loaders_onnx` (see `camerachatbot/runtime/
        # bootstrap.py`), so it always owns its own ONNX Runtime session and
        # needs neither `.to()` nor `.fuse()`.

        self.class_map = (class_map or {
            "sit": "sentado", "sitting": "sentado", "sentado": "sentado",
            "stand": "de pie", "standing": "de pie", "parado": "de pie", "de pie": "de pie"
        })
        self.names = self.model.names

    def _map_label(self, raw_label: str) -> str:
        key = str(raw_label).strip().lower()
        return self.class_map.get(key, raw_label)

    def run_on_json(self, json_file: str, topk_per_frame: int = 6, min_area_ratio: float = 0.04) -> str:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        crops, backrefs = [], []
        # Recorre frames y selecciona Top-K personas por área (y filtra muy pequeñas)
        for frame_id, entries in data.items():
            if "neighborhood" in frame_id:
                continue  # Evita procesar ese frame fantasma

            frame_path = os.path.join(self.frames_folder, f"{frame_id}.jpg")
            img = cv2.imread(frame_path)
            if img is None:
                print(f"[PoseCls] WARN: no se encontró frame '{frame_id}' en {self.frames_folder}")
                continue

            H, W = img.shape[:2]
            frame_area = float(H * W)

            # índices de personas en este frame
            person_idxs = [i for i, p in enumerate(entries) if p.get("kind") == "person"]

            # ordenar por área descendente
            def _area(p):
                x1, y1, x2, y2 = map(int, p["bbox"])
                return max(0, (x2 - x1)) * max(0, (y2 - y1))

            person_idxs.sort(key=lambda i: _area(entries[i]), reverse=True)

            # quedarnos con Top-K
            if topk_per_frame is not None and topk_per_frame > 0:
                person_idxs = person_idxs[:topk_per_frame]

            for i in person_idxs:
                p = entries[i]
                x1, y1, x2, y2 = map(int, p["bbox"])
                x1 = max(0, min(x1, W - 1));
                x2 = max(0, min(x2, W))
                y1 = max(0, min(y1, H - 1));
                y2 = max(0, min(y2, H))
                area = max(0, (x2 - x1)) * max(0, (y2 - y1))

                p.setdefault("attributes", {})
                # gating por área mínima relativa
                if area <= 0 or (area / (frame_area + 1e-6)) < float(min_area_ratio):
                    p["attributes"]["pose"] = p["attributes"].get("pose", "indefinido")
                    p["attributes"]["pose_conf"] = p["attributes"].get("pose_conf", 0.0)
                    p["attributes"]["pose_source"] = "yolo-cls"
                    continue

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    p["attributes"]["pose"] = p["attributes"].get("pose", "indefinido")
                    p["attributes"]["pose_conf"] = p["attributes"].get("pose_conf", 0.0)
                    p["attributes"]["pose_source"] = "yolo-cls"
                    continue

                crops.append(crop)
                backrefs.append((frame_id, i))

        # Si no hay crops, guardar tal cual y salir
        out_file = os.path.join(os.path.dirname(json_file), "tracking_pose_2.json")
        if not crops:
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"[PoseCls] (sin crops) JSON guardado en {out_file}")
            return out_file

        # Inferencia en batch. `PoseClsOnnxClassifier.predict()` ya devuelve
        # `(pred_idx, conf)` por crop — batching/precisión/softmax son su
        # propia responsabilidad (ver onnx_pose_classifier.py). Fase 3c
        # removed the legacy torch/ultralytics `.probs`-parsing path this
        # replaced — see git history (pre-Fase-3c) if it's ever needed again.
        names = getattr(self, "names", None)
        conf_thr = float(getattr(self, "confs", 0.60))
        preds = self.model.predict(crops)

        # Escribe resultados de vuelta
        for (frame_id, i), (pred_idx, conf) in zip(backrefs, preds):
            p = data[frame_id][i]
            p.setdefault("attributes", {})

            # mapea etiqueta cruda → etiqueta final
            if isinstance(names, dict):
                raw = names.get(pred_idx, str(pred_idx))
            elif isinstance(names, (list, tuple)) and pred_idx < len(names):
                raw = names[pred_idx]
            else:
                raw = str(pred_idx)

            label = self._map_label(raw)  # p.ej. {"sit":"sentado","stand":"de pie", ...}

            if conf >= conf_thr:
                p["attributes"]["pose"] = label
                p["attributes"]["pose_conf"] = conf
            else:
                p["attributes"]["pose"] = p["attributes"].get("pose", "indefinido")
                p["attributes"]["pose_conf"] = conf

            p["attributes"]["pose_source"] = "yolo-cls"

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"[PoseCls] JSON con pose (YOLO-CLS) guardado en {out_file}")
        return out_file
