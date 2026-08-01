"""
contextual_neighborhood.py
Detecta objetos/personas con YOLOv8, asigna IDs a personas y construye un
'contextual neighborhood' con relaciones (derecha/izquierda/cerca/encima/abajo/solapa).
Funciona con imagen única o con secuencia (si alimentas frames sucesivos).
"""

import math
import json
import numpy as np
import cv2
from ultralytics import YOLO

# Optional tracker (for multi-frame). If no tracker disponible, usamos IDs por frame.
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    HAS_DEEPSORT = True
except Exception:
    HAS_DEEPSORT = False

# -----------------------------
# Configuración (ajusta aquí)
# -----------------------------
YOLO_MODEL = "../models/yolov10m.pt"   # cambia si quieres otro
DETECTION_CONF = 0.35       # confidence threshold
IOU_THRESHOLD = 0.45        # NMS IoU
PROXIMITY_FACTOR = 0.8      # persona cerca si distancia centro < PROXIMITY_FACTOR * diag_bbox_person
PERSON_CLASS_ID = 0         # COCO: person = 0
OBJECT_CLASSES_KEEP = None  # None => keep all classes excepto 'person' en objetos (puedes filtrar)
USE_TRACKER = True          # usar DeepSort si HAS_DEEPSORT True y si procesas video/secuencia

# -----------------------------
# Utilidades geométricas
# -----------------------------
def bbox_to_xywh(bbox):
    # bbox = [x1, y1, x2, y2]
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w/2.0
    cy = y1 + h/2.0
    return (cx, cy, w, h)

def bbox_diagonal(bbox):
    x1, y1, x2, y2 = bbox
    return math.hypot(x2 - x1, y2 - y1)

def euclidean(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def relative_position(person_c, obj_c, x_margin_ratio=0.15):
    # person_c, obj_c: (cx, cy, w, h) or center coords
    px, py = person_c[0], person_c[1]
    ox, oy = obj_c[0], obj_c[1]
    # direction simple
    dx = px - ox
    dy = py - oy
    angle = math.degrees(math.atan2(dy, dx))  # -180..180, 0 = object -> person horizontally to right
    # left/right:
    if abs(dx) > abs(dy):
        if dx > 0:
            horiz = "right_of"   # person is to right of object
        else:
            horiz = "left_of"
    else:
        horiz = None
    # up/down based on y (image coords: y increases downward)
    if abs(dy) > abs(dx):
        if dy > 0:
            vert = "below"    # person is below object (i.e., person y > object y)
        else:
            vert = "above"
    else:
        vert = None
    return {"dx": dx, "dy": dy, "angle_deg": angle, "horiz": horiz, "vert": vert}

# -----------------------------
# Core: detección + vecindario
# -----------------------------
class ContextualNeighborhood:
    def __init__(self, model_path=YOLO_MODEL):
        self.model = YOLO(model_path)
        self.tracker = None
        if HAS_DEEPSORT and USE_TRACKER:
            # DeepSort init con configuración por defecto suficiente para demo
            self.tracker = DeepSort(max_age=30)

        self.next_person_id = 1

    def run_on_image(self, image_bgr):
        """Procesa una sola imagen y devuelve (annotated_image, neighborhood_json)."""
        h, w = image_bgr.shape[:2]
        # YOLOv8 inference
        results = self.model.predict(source=image_bgr, imgsz=max(h,w),
                                     conf=DETECTION_CONF, iou=IOU_THRESHOLD, verbose=False)[0]
        # results.boxes.xyxy, results.boxes.conf, results.boxes.cls
        boxes = results.boxes.xyxy.cpu().numpy() if hasattr(results.boxes, 'xyxy') else np.array([])
        scores = results.boxes.conf.cpu().numpy() if hasattr(results.boxes, 'conf') else np.array([])
        classes = results.boxes.cls.cpu().numpy().astype(int) if hasattr(results.boxes, 'cls') else np.array([])

        # Build detections list
        detections = []
        for bbox, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(float, bbox)
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "score": float(score),
                "class_id": int(cls)
            })

        # Separate persons and objects
        persons = []
        objects = []
        for det in detections:
            if det["class_id"] == PERSON_CLASS_ID:
                persons.append(det)
            else:
                if OBJECT_CLASSES_KEEP is None or det["class_id"] in OBJECT_CLASSES_KEEP:
                    objects.append(det)

        # Assign IDs to persons. For single image we just give unique ID per person.
        # If tracker available and you call run_on_image sequentially with frames, you should
        # feed detections to tracker to get persistent IDs.
        if self.tracker is not None:
            # Build detections for tracker: (bbox, score, class_name)
            ds_dets = []
            for det in persons:
                x1, y1, x2, y2 = det["bbox"]
                ds_dets.append(([x1, y1, x2, y2], det["score"], "person"))
            tracks = self.tracker.update_tracks(ds_dets, frame=image_bgr)
            # tracks -> list of Track objects with track_id, to_tlbr()
            persons_with_id = []
            for tr in tracks:
                if not tr.is_confirmed():
                    continue
                tlbr = tr.to_tlbr()  # (minx,miny,maxx,maxy)
                persons_with_id.append({
                    "id": int(tr.track_id),
                    "bbox": [float(tlbr[0]), float(tlbr[1]), float(tlbr[2]), float(tlbr[3])]
                })
        else:
            # Fallback: assign incremental IDs per detection in this frame
            persons_with_id = []
            for det in persons:
                pid = self.next_person_id
                self.next_person_id += 1
                persons_with_id.append({
                    "id": pid,
                    "bbox": det["bbox"]
                })

        # Build contextual neighborhood: for each person, find nearby objects and persons
        neighborhood = {"persons": []}

        # Precompute centers & diags
        for p in persons_with_id:
            p_cx, p_cy, p_w, p_h = bbox_to_xywh(p["bbox"])
            p_diag = bbox_diagonal(p["bbox"])
            p_center = (p_cx, p_cy)
            # Find object neighbors
            obj_neighbors = []
            for i, obj in enumerate(objects):
                o_cx, o_cy, o_w, o_h = bbox_to_xywh(obj["bbox"])
                o_center = (o_cx, o_cy)
                dist = euclidean(p_center, o_center)
                rel = relative_position((p_cx, p_cy, p_w, p_h), (o_cx, o_cy))
                relation_types = []
                # proximity
                if dist < PROXIMITY_FACTOR * p_diag:
                    relation_types.append("near")
                # left/right/above/below if applicable
                if rel["horiz"]:
                    relation_types.append(rel["horiz"])
                if rel["vert"]:
                    relation_types.append(rel["vert"])
                # overlap
                # compute IoU
                x1 = max(p["bbox"][0], obj["bbox"][0])
                y1 = max(p["bbox"][1], obj["bbox"][1])
                x2 = min(p["bbox"][2], obj["bbox"][2])
                y2 = min(p["bbox"][3], obj["bbox"][3])
                inter_area = max(0, x2-x1) * max(0, y2-y1)
                a_p = (p["bbox"][2]-p["bbox"][0])*(p["bbox"][3]-p["bbox"][1])
                a_o = (obj["bbox"][2]-obj["bbox"][0])*(obj["bbox"][3]-obj["bbox"][1])
                union = a_p + a_o - inter_area
                iou = inter_area/union if union>0 else 0.0
                if iou > 0.05:
                    relation_types.append("overlap")
                obj_neighbors.append({
                    "object_index": i,
                    "class_id": obj["class_id"],
                    "bbox": obj["bbox"],
                    "distance_px": float(dist),
                    "relation": list(dict.fromkeys(relation_types)),  # unique
                    "angle_deg": rel["angle_deg"],
                    "iou": float(iou)
                })

            # Find person neighbors
            person_neighbors = []
            for other in persons_with_id:
                if other["id"] == p["id"]:
                    continue
                o_cx, o_cy, o_w, o_h = bbox_to_xywh(other["bbox"])
                dist = euclidean((p_cx, p_cy), (o_cx, o_cy))
                avg_diag = (p_diag + bbox_diagonal(other["bbox"])) / 2.0
                near = dist < PROXIMITY_FACTOR * avg_diag
                rel = relative_position((p_cx, p_cy, p_w, p_h), (o_cx, o_cy))
                relation_types = []
                if near:
                    relation_types.append("near_person")
                if rel["horiz"]:
                    relation_types.append(rel["horiz"])
                if rel["vert"]:
                    relation_types.append(rel["vert"])
                person_neighbors.append({
                    "person_id": other["id"],
                    "bbox": other["bbox"],
                    "distance_px": float(dist),
                    "relation": relation_types,
                    "angle_deg": rel["angle_deg"]
                })

            neighborhood["persons"].append({
                "person_id": p["id"],
                "bbox": p["bbox"],
                "center": [p_cx, p_cy],
                "diag": p_diag,
                "objects": obj_neighbors,
                "person_neighbors": person_neighbors
            })

        # Optional: annotated image
        annotated = image_bgr.copy()
        # draw objects (different color for person vs object)
        # COLORS simple
        for entry in neighborhood["persons"]:
            pid = entry["person_id"]
            x1,y1,x2,y2 = map(int, entry["bbox"])
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,200,0), 2)
            cv2.putText(annotated, f"P{pid}", (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)

            # draw lines to nearby objects
            for obj in entry["objects"]:
                ox1,oy1,ox2,oy2 = map(int, obj["bbox"])
                cx_p = int(entry["center"][0]); cy_p = int(entry["center"][1])
                o_cx = int((ox1+ox2)/2); o_cy = int((oy1+oy2)/2)
                # line
                cv2.line(annotated, (cx_p,cy_p), (o_cx,o_cy), (200,100,0), 1)
                # label with relation
                rel_txt = ",".join(obj["relation"][:2]) if obj["relation"] else ""
                cv2.putText(annotated, rel_txt, (o_cx+4,o_cy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,100,0), 1)

        # draw objects
        for i, obj in enumerate(objects):
            x1,y1,x2,y2 = map(int, obj["bbox"])
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (180,50,200), 2)
            cv2.putText(annotated, f"O{i}:{obj['class_id']}", (x1,y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,50,200), 1)

        return annotated, neighborhood

    def build_db_json(self, neighborhood, frame_id=0, image_shape=None):
        """Convierte el neighborhood JSON a formato listo para la BD."""
        h, w = image_shape[:2]
        rows = []
        rid = 1
        for person in neighborhood["persons"]:
            parent_id = person["person_id"]
            # Persona ↔ Objetos
            for obj in person["objects"]:
                dx = obj["bbox"][0] - person["bbox"][0]
                dy = obj["bbox"][1] - person["bbox"][1]
                rel_txt = ",".join(obj["relation"]) if obj["relation"] else None
                if not rel_txt:
                    continue
                rows.append({
                    "id": rid,
                    "key_frame_id": frame_id,
                    "related_object_id": obj["object_index"],
                    "intersection": round(obj["iou"], 4) if obj["iou"] > 0 else None,
                    "x_alignment": round(obj["angle_deg"] / 180.0, 4),  # o normalizado si prefieres
                    "y_alignment": None,  # puedes agregar si usas dy/h
                    "relation": rel_txt,
                    "markdown": f"**Persona {parent_id}** está *{rel_txt.replace(',', ' y ')}* del objeto {obj['object_index']}.",
                    "parent_object_id": parent_id
                })
                rid += 1

            # Persona ↔ Persona
            for p2 in person["person_neighbors"]:
                rel_txt = ",".join(p2["relation"]) if p2["relation"] else None
                if not rel_txt:
                    continue
                rows.append({
                    "id": rid,
                    "key_frame_id": frame_id,
                    "related_object_id": p2["person_id"],
                    "intersection": None,
                    "x_alignment": round(p2["angle_deg"] / 180.0, 4),
                    "y_alignment": None,
                    "relation": rel_txt,
                    "markdown": f"**Persona {parent_id}** está *{rel_txt.replace(',', ' y ')}* de **Persona {p2['person_id']}**.",
                    "parent_object_id": parent_id
                })
                rid += 1
        return rows

# -----------------------------
# Ejemplo de uso (función main)
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Ruta a imagen")
    parser.add_argument("--out", default="annotated.jpg", help="Imagen anotada salida")
    parser.add_argument("--model", default=YOLO_MODEL, help="Ruta modelo YOLO")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    cn = ContextualNeighborhood(model_path=args.model)
    annotated, neighborhood = cn.run_on_image(img)
    db_json = cn.build_db_json(neighborhood, frame_id=1, image_shape=img)

    print(json.dumps(db_json, indent=2))

    # Guardar salida
    cv2.imwrite(args.out, annotated)
    print("Contextual neighborhood (JSON):")
    print(json.dumps(neighborhood, indent=2))
    print(f"Imagen anotada guardada en: {args.out}")
