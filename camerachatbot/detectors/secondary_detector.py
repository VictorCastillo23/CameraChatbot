"""Fase 6 (PR10): the secondary/allowlisted-object detail detector.

`AllowlistedDetector` implements the same detail-detector contract every
other detector in `pipeline_service.build_detail_detectors()` already
follows (a `frames_folder` attribute set externally by the orchestrator
loop, plus `run_on_json(path) -> path`), wrapping `YOLOOnnxDetector`
(Fase 3a, `camerachatbot/detectors/onnx_yolo.py`) to append `kind="object"`
entries for a configurable class allowlist (`security_config.WEAPONS_DETECTOR`).

Unconfigured (`model_path=None`, the shipped default) it stays a genuine
no-op: no `YOLOOnnxDetector` session is ever created, and `run_on_json()`
returns its input path unchanged.

`YOLOOnnxDetector` is imported at module level (not lazily inside
`__init__`) so it resolves as a plain module global at call time,
matching this codebase's established monkeypatch convention for
class-shaped seams (see `orchestrator.py`'s `AuthorizationRegistry`
import, patched the same way by `test_identity_thresholds.py`). This is
safe cost-wise: `onnxruntime` is already an unconditional dependency of
the whole pipeline since Fase 3b (`bootstrap.py` always loads via
`loaders_onnx`), and importing the class itself does not create an ONNX
Runtime session -- that only happens in the configured branch of
`__init__` below.
"""

import json
import os

import cv2

from camerachatbot.detectors.onnx_yolo import YOLOOnnxDetector


class AllowlistedDetector:
    def __init__(self, model_path=None, class_names=None, conf=0.35, providers=None):
        self.model_path = model_path
        self.conf = float(conf)
        self.class_names = set(class_names) if class_names else set()
        # Set externally by the orchestrator loop, like every other detail
        # detector -- not required at construction time.
        self.frames_folder = None

        if model_path is None:
            self._enabled = False
            print(
                "[AllowlistedDetector] ADVERTENCIA: no se configuro 'model_path' "
                "en WEAPONS_DETECTOR; el detector queda inactivo (no-op)."
            )
        else:
            self.model = YOLOOnnxDetector(model_path, providers=providers, conf=self.conf)
            self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def run_on_json(self, json_file: str) -> str:
        if not self._enabled:
            return json_file

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for frame_id, entries in data.items():
            if "neighborhood" in frame_id:
                continue  # Evita procesar ese frame fantasma

            frame_path = os.path.join(self.frames_folder, f"{frame_id}.jpg")
            img = cv2.imread(frame_path)
            if img is None:
                print(
                    f"[AllowlistedDetector] WARN: no se encontro frame "
                    f"'{frame_id}' en {self.frames_folder}"
                )
                continue

            results = self.model(img)
            result = results[0]
            for b in result.boxes:
                class_name = result.names[int(b.cls[0])]
                if class_name not in self.class_names:
                    continue
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                entries.append({
                    "kind": "object",
                    "class_name": class_name,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(b.conf[0]),
                    "attributes": {},
                })

        out_file = os.path.join(os.path.dirname(json_file), "tracking_weapons_6.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"[AllowlistedDetector] JSON con objetos guardado en {out_file}")
        return out_file
