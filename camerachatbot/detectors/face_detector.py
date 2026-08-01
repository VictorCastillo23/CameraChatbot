import os, cv2, json,numpy as np,mediapipe as mp
from typing import Dict, List,Any

class FaceDetector():
    def __init__(self, frames_folder: str = None, res: List[Any] = None,
                 max_num_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 ear_thresh: float = 0.21,
                 mouth_thresh: float = 0.05,
                 gaze_thresh: float = 0.02):
        self.frames_folder = frames_folder
        self.res = res
        self._frames_map = None  # cache del mixin

        self.ear_thresh = ear_thresh
        self.mouth_thresh = mouth_thresh
        self.gaze_thresh = gaze_thresh

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence
        )

        self.left_eye_idxs = [33, 160, 158, 133, 153, 144, 163, 7, 246]
        self.right_eye_idxs = [362, 385, 387, 263, 373, 380, 390, 249, 466]

        self.iris_left_idx = 468
        self.iris_right_idx = 473

    @staticmethod
    def _euclid_norm(p, q):
        return float(np.linalg.norm(np.asarray(p) - np.asarray(q)))

    def eye_aspect_ratio(self, landmarks, eye_idxs):
        # EAR en coordenadas normalizadas del recorte
        pts = [(landmarks[i].x, landmarks[i].y) for i in eye_idxs]
        p1, p2, p3, p4, p5, p6 = pts  # (x,y) normalizados
        vertical1 = self._euclid_norm(p2, p6)
        vertical2 = self._euclid_norm(p3, p5)
        horizontal = self._euclid_norm(p1, p4) + 1e-9
        return (vertical1 + vertical2) / (2.0 * horizontal)

    def mouth_open_ratio(self, landmarks, top_idx, bottom_idx):
        top = (landmarks[top_idx].x, landmarks[top_idx].y)
        bottom = (landmarks[bottom_idx].x, landmarks[bottom_idx].y)
        # distancia vertical normalizada al recorte
        return self._euclid_norm(top, bottom)

    def classify_face(self, landmarks) -> Dict[str, Any]:
        mouth_ratio = self.mouth_open_ratio(landmarks, 13, 14)
        mouth_state = "abierta" if mouth_ratio > self.mouth_thresh else "cerrada"

        iris_left = (landmarks[self.iris_left_idx].x, landmarks[self.iris_left_idx].y)
        iris_right = (landmarks[self.iris_right_idx].x, landmarks[self.iris_right_idx].y)

        left_eye_pts = [(landmarks[i].x, landmarks[i].y) for i in self.left_eye_idxs]
        right_eye_pts = [(landmarks[i].x, landmarks[i].y) for i in self.right_eye_idxs]

        def eye_geometry(eye_pts):
            xs = [p[0] for p in eye_pts]
            ys = [p[1] for p in eye_pts]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            w = max(max(xs) - min(xs), 1e-6)
            h = max(max(ys) - min(ys), 1e-6)
            return cx, cy, w, h

        cx_l, cy_l, w_l, h_l = eye_geometry(left_eye_pts)
        cx_r, cy_r, w_r, h_r = eye_geometry(right_eye_pts)

        dx_l = (iris_left[0] - cx_l) / w_l
        dy_l = (iris_left[1] - cy_l) / h_l
        dx_r = (iris_right[0] - cx_r) / w_r
        dy_r = (iris_right[1] - cy_r) / h_r

        dx = (dx_l + dx_r) / 2.0
        dy = (dy_l + dy_r) / 2.0

        t_center_x, t_limit_x = getattr(self, "gaze_t1x", 0.05), getattr(self, "gaze_t3x", 0.25)
        t_center_y, t_limit_y = getattr(self, "gaze_t1y", 0.05), getattr(self, "gaze_t3y", 0.20)

        def axis_score(abs_val, t_center, t_limit):
            if abs_val <= t_center:
                return 1.0
            elif abs_val >= t_limit:
                return 0.0
            else:
                return 1.0 - (abs_val - t_center) / (t_limit - t_center)

        score_x = axis_score(abs(dx), t_center_x, t_limit_x)
        score_y = axis_score(abs(dy), t_center_y, t_limit_y)

        attention = 0.7 * score_x + 0.3 * score_y

        prev_attention = getattr(self, "_prev_attention", attention)
        attention = 0.8 * prev_attention + 0.2 * attention

        score = round(float(max(0.0, min(1.0, attention))), 3)
        #score = 1 if score <= 0.7 else 0
        return {
            "attention": {
               "attention": round(float(score), 3),
            },
            "mouth": {
                "state": mouth_state,
                "ratio": round(float(mouth_ratio), 3)
            }
        }

    def run_on_json(self, json_file: str) -> str:
        with open(json_file, "r", encoding="utf-8") as f:
            results_data = json.load(f)

        updated_data: Dict[str, List[Dict[str, Any]]] = {}

        for frame_id, entries in results_data.items():
            if "neighborhood" in frame_id:
                continue

            updated_entries = []
            frame_path = os.path.join(self.frames_folder, f"{frame_id}.jpg")
            img = cv2.imread(frame_path)
            if img is None:
                print(f"[Face] WARN: no se encontró frame '{frame_id}' ni en res ni en {self.frames_folder}")
                results_data[frame_id] = entries
                continue

            h, w = img.shape[:2]

            for person in entries:
                # siempre preserva entradas no-persona
                if person.get('kind') != 'person':
                    updated_entries.append(person)
                    continue

                person.setdefault("attributes", {})

                # recorte robusto
                x1, y1, x2, y2 = map(int, person["bbox"])
                x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h))
                if x2 <= x1 or y2 <= y1:
                    # bbox inválido → marca vacío
                    person["attributes"]["face"] = {"eyes": None, "mouth": None}
                    updated_entries.append(person)
                    continue

                crop_bgr = img[y1:y2, x1:x2]
                rgb_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

                # MediaPipe FaceMesh opera en coords normalizadas al recorte
                results = self.face_mesh.process(rgb_crop)

                if results.multi_face_landmarks:
                    # toma la primera cara (o podrías elegir la más centrada)
                    lms = results.multi_face_landmarks[0].landmark
                    face_info = self.classify_face(lms)

                    h_crop, w_crop = crop_bgr.shape[:2]
                    xs = [max(0.0, min(1.0, lm.x)) for lm in lms]
                    ys = [max(0.0, min(1.0, lm.y)) for lm in lms]

                    fx1 = x1 + int(min(xs) * w_crop)
                    fy1 = y1 + int(min(ys) * h_crop)
                    fx2 = x1 + int(max(xs) * w_crop)
                    fy2 = y1 + int(max(ys) * h_crop)

                    fx1 = max(0, min(fx1, w - 1))
                    fx2 = max(0, min(fx2, w))
                    fy1 = max(0, min(fy1, h - 1))
                    fy2 = max(0, min(fy2, h))

                    face_info["bbox"] = [fx1, fy1, fx2, fy2]
                else:
                    face_info = {"eyes": None, "mouth": None,"bbox":None}

                person["attributes"]["face"] = face_info
                updated_entries.append(person)

            results_data[frame_id] = updated_entries
            print(f"[Face] Frame {frame_id} procesado "
                  f"(personas: {sum(1 for e in updated_entries if e.get('kind')=='person')}, "
                  f"objetos: {sum(1 for e in updated_entries if e.get('kind')=='object')})")

        output_file = os.path.join(os.path.dirname(json_file), "tracking_face_3.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=4, ensure_ascii=False)

        print(f"[OK] JSON con cara/ojos guardado en {output_file}")
        return output_file


'''
    def classify_face(self, landmarks) -> Dict[str, Any]:
        """
        Retorna:
        {
          "eyes": {
            "state": "abiertos|cerrados",
            "ear": 0.123,
            "gaze": {
              "label": "mirando a cámara|casi a cámara|desviado|muy desviado",
              "direction": "izquierda|derecha|centro",
              "deviation": 0.042,   # |dx| normalizado por ancho del ojo (0=centro)
              "score": 0.91          # 1=perfectamente centrado, 0=muy desviado
            }
          },
          "mouth": { "state": "abierta|cerrada", "ratio": 0.123 }
        }
        """

        # --- OJOS: EAR y estado
        left_ear = self.eye_aspect_ratio(landmarks, self.left_eye_idxs)
        right_ear = self.eye_aspect_ratio(landmarks, self.right_eye_idxs)
        avg_ear = (left_ear + right_ear) / 2.0
        eyes_state = "cerrados" if avg_ear < self.ear_thresh else "abiertos"

        # --- BOCA
        mouth_ratio = self.mouth_open_ratio(landmarks, 13, 14)  # MP lips sup/inf
        mouth_state = "abierta" if mouth_ratio > self.mouth_thresh else "cerrada"

        # --- GAZE: qué tanto mira a cámara (horizontal)
        # Usamos sólo el ojo derecho + su iris (como venías haciendo) pero normalizamos por el ancho del ojo.
        iris = (landmarks[self.iris_right_idx].x, landmarks[self.iris_right_idx].y)

        # Centro del ojo derecho (promedio de puntos del ojo)
        right_eye_pts = [(landmarks[i].x, landmarks[i].y) for i in self.right_eye_idxs]
        eye_center_x = float(sum(p[0] for p in right_eye_pts) / len(right_eye_pts))

        # Ancho del ojo derecho (max_x - min_x) para normalizar dx
        min_x = min(p[0] for p in right_eye_pts)
        max_x = max(p[0] for p in right_eye_pts)
        eye_width = max(max_x - min_x, 1e-6)  # evita división por cero

        # dx normalizado: cuántas "anchuras de ojo" se desplaza la pupila desde el centro
        dx_norm = float((iris[0] - eye_center_x) / eye_width)  # ~[-0.5, 0.5] aprox
        abs_dx = abs(dx_norm)

        # Umbrales graduados (puedes moverlos a self si quieres tunear en runtime)
        t1 = getattr(self, "gaze_t1", 0.06)  # <= t1  => mirando a cámara
        t2 = getattr(self, "gaze_t2", 0.12)  # <= t2  => casi a cámara
        t3 = getattr(self, "gaze_t3", 0.20)  # <= t3  => desviado (más de esto = muy desviado)

        if abs_dx <= t1:
            gaze_label = "mirando a cámara"
        elif abs_dx <= t2:
            gaze_label = "casi a cámara"
        elif abs_dx <= t3:
            gaze_label = "desviado"
        else:
            gaze_label = "muy desviado"

        # Dirección (útil cuando no está centrado)
        if abs_dx <= t1:
            direction = "centro"
        else:
            direction = "derecha" if dx_norm > 0 else "izquierda"

        # Score de alineación con la cámara (1 centrado, 0 muy desviado):
        # Decrece linealmente desde t1 hasta t3 y se satura.
        # Dentro de t1 forzamos 1.0 para premiar el centro.
        if abs_dx <= t1:
            score = 1.0
        elif abs_dx >= t3:
            score = 0.0
        else:
            # mapear [t1, t3] -> [1, 0]
            score = 1.0 - (abs_dx - t1) / (t3 - t1)
            """
            "eyes": {
                "state": eyes_state,
                "ear": round(float(avg_ear), 3),
                "gaze": {
                    "label": gaze_label,
                    "direction": direction,
                    "deviation": round(float(abs_dx), 3),
                    "score": round(float(score), 3)
                }
            },
            """
        return {
            "attention": round(float(score), 3),
            "mouth": {
                "state": mouth_state,
                "ratio": round(float(mouth_ratio), 3)
            }
        }


'''

