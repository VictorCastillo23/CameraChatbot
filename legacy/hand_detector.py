import os, cv2, json,mediapipe as mp
from typing import Dict, List,Any

class HandDetector():
    def __init__(self, frames_folder: str = None, res: List[Any] = None,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.2):
        self.frames_folder = frames_folder
        self.res = res
        self._frames_map = None

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=max_num_hands,
            min_tracking_confidence=0.3,
            min_detection_confidence=min_detection_confidence
        )

    def classify_gesture(self, landmarks, hand_label: str):

        WRIST = 0
        THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
        tips = [4, 8, 12, 16, 20]
        pip_of = {8: 6, 12: 10, 16: 14, 20: 18}

        finger_states = []
        for tip in tips[1:]:
            pip = pip_of[tip]
            finger_states.append(1 if landmarks[tip].y < landmarks[pip].y else 0)

        if hand_label == "Right":
            thumb_extended_side = 1 if landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x else 0
        else:
            thumb_extended_side = 1 if landmarks[THUMB_TIP].x > landmarks[THUMB_IP].x else 0
        finger_states.insert(0, thumb_extended_side)

        tol_fold = 0.02
        tol_up = 0.02
        others_folded = all(
            landmarks[tip].y >= landmarks[pip_of[tip]].y - tol_fold
            for tip in [8, 12, 16, 20]
        )
        thumb_up_vertical = (
                (landmarks[THUMB_TIP].y + tol_up < landmarks[THUMB_IP].y) and
                (landmarks[THUMB_TIP].y + tol_up < landmarks[THUMB_MCP].y) and
                (landmarks[THUMB_TIP].y + tol_up < landmarks[WRIST].y)
        )
        thumb_extended = (thumb_extended_side == 1)

        if others_folded and thumb_up_vertical and thumb_extended:
            return "pulgar_arriba"

        total = sum(finger_states)
        if total == 0:
            return "puño"
        elif total == 5:
            return "palma"
        else:
            return "otro"

    def run_on_json(self, json_file: str) -> str:
        with open(json_file, "r", encoding="utf-8") as f:
            results_data = json.load(f)

        updated_data: Dict[str, List[Dict[str, Any]]] = {}

        for frame_id, entries in results_data.items():

            if frame_id == 'neighborhood':
                continue
            updated_entries: List[Dict[str, Any]] = []
            frame_path = os.path.join(self.frames_folder, f"{frame_id}.jpg")
            img = cv2.imread(frame_path)
            if img is None:
                print(f"[Hands] WARN: no se encontró frame '{frame_id}' ni en res ni en {self.frames_folder}")
                results_data[frame_id] = entries
                continue

            h, w = img.shape[:2]

            for person in entries:
                if person.get('kind') != 'person':
                    updated_entries.append(person)
                    continue

                person.setdefault("attributes", {})

                x1, y1, x2, y2 = map(int, person["bbox"])
                #print(f'bbox de {person["confidence"]} : {person["bbox"]}')
                x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h))
                if x2 <= x1 or y2 <= y1:
                    person["attributes"]["hands"] = None
                    updated_entries.append(person)
                    continue

                crop_bgr = img[y1:y2, x1:x2]

                if crop_bgr.size == 0:
                    person["attributes"]["hands"] = None
                    print(f'No se encontraron manos en la persona con confidence{person["confidence"]}')
                    updated_entries.append(person)
                    continue

                rgb_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

                results = self.hands.process(rgb_crop)

                if person["confidence"] == 0.6870084404945374:
                    print("rgb_crop dtype:", rgb_crop.dtype, "shape:", rgb_crop.shape)
                    #print("Results:", results)
                    #print("pid:", results)
                    #cv2.imshow("Crop", cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR))
                    #cv2.waitKey(0)

                hand_info_list = []
                if results.multi_hand_landmarks and results.multi_handedness:
                    num_hands = len(results.multi_hand_landmarks)
                    #print(f"[INFO] Manos detectadas: {num_hands} en la persona con confidence{person['confidence']}")

                    for idx, (hand_landmarks, handedness) in enumerate(
                            zip(results.multi_hand_landmarks, results.multi_handedness)
                    ):
                        hand_label = handedness.classification[0].label
                        #print(f"  ↳ Mano #{idx + 1}: {hand_label}")

                        # Mostrar coordenadas resumidas
                        coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                        #print(f"     - Total puntos: {len(coords)}")
                        #print(f"     - Primer punto (dedo pulgar base): {coords[0]}")

                        # Clasificar gesto
                        gesture = self.classify_gesture(hand_landmarks.landmark, hand_label)
                        #print(f"     - Gesto detectado: {gesture}")

                        hand_info = {
                            "hand": hand_label,
                            "gesture": gesture
                        }
                        hand_info_list.append(hand_info)

                    # Confirmar que se guardó algo en la lista
                    #print(f"[INFO] Total info de manos guardada: {len(hand_info_list)} elementos")
                    #for i, info in enumerate(hand_info_list):
                        #print(f"   {i + 1}. Mano: {info['hand']}, Gesto: {info['gesture']}")

                #else:
                    #print(f"[INFO] No se detectaron manos o falta handedness. en la persona con confidence{person['confidence']}")

                person["attributes"]["hands"] = hand_info_list if hand_info_list else None
                updated_entries.append(person)

            results_data[frame_id] = updated_entries
            print(f"[Hands] Frame {frame_id} procesado "
                  f"(personas: {sum(1 for e in updated_entries if e.get('kind')=='person')}, "
                  f"objetos: {sum(1 for e in updated_entries if e.get('kind')=='object')})")

        output_file = os.path.join(os.path.dirname(json_file), "tracking_hands_4.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=4, ensure_ascii=False)

        print(f"[OK] JSON con manos/gestos guardado en {output_file}")
        return output_file
