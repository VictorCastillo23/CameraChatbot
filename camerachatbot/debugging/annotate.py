import os, json, glob
import cv2
import numpy as np

from camerachatbot import paths

# ---------- utilidades de dibujo ----------
def _ensure_int_bbox(b):
    x1, y1, x2, y2 = map(int, b)
    return x1, y1, x2, y2

def _put_multiline(img, origin, lines, color=(255,255,255), bg=(0,0,0),
                   scale=0.5, thickness=1, pad=2, line_h=None):
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_h = line_h or int(16 * scale + 6)
    for line in lines:
        line = "" if line is None else str(line)
        (tw, th), _ = cv2.getTextSize(line, font, scale, thickness)
        cv2.rectangle(img, (x, y - th - pad), (x + tw + pad*2, y + pad), bg, -1)
        cv2.putText(img, line, (x + pad, y - pad), font, scale, color, thickness, cv2.LINE_AA)
        y += line_h

def _fmt_float(x, n=3):
    try: return f"{float(x):.{n}f}"
    except: return str(x)

def _find_image_for_frame(images_dir, frame_id, exts=(".jpg",".png",".jpeg",".bmp",".webp")):
    for ext in exts:
        p = os.path.join(images_dir, f"{frame_id}{ext}")
        if os.path.exists(p):
            return p
    for ext in exts:
        g = glob.glob(os.path.join(images_dir, f"*{frame_id}*{ext}"))
        if g:
            return g[0]
    return None

def _build_person_lines(entry):
    lines = []
    attrs = entry.get("attributes", {}) or {}
    pose = attrs.get("pose")
    pose_conf = attrs.get("pose_conf")
    if pose is not None:

        lines.append(f"pose: {pose}" + (f" ({_fmt_float(pose_conf)})" if pose_conf is not None else ""))

    if entry.get("track_id") is not None:
        lines.append(f"track: {entry['track_id']}")
    if entry.get("person_global_id") is not None:
        lines.append(f"pid: {entry['person_global_id']}")
    return lines

def _build_depth_lines(depth):
    if not depth:
        return []
    return [
        f"dzμ:{_fmt_float(depth.get('mean'))}",
        f"med:{_fmt_float(depth.get('median'))}",
        f"min:{_fmt_float(depth.get('min'))}",
    ]

def _build_face_panel_lines(face):
    if not isinstance(face, dict):
        return []

    lines = []
    eyes = face.get("eyes") or {}
    mouth = face.get("mouth") or {}
    emo = face.get("emotion") or {}

    if eyes:
        parts = []
        if "state" in eyes:
            parts.append(f"eyes:{eyes['state']}")
        if "ear" in eyes:
            parts.append(f"EAR:{_fmt_float(eyes['ear'])}")
        if "gaze" in eyes:
            parts.append(f"gaze:{eyes['gaze']}")
        if parts:
            lines.append(" ".join(parts))

    if mouth:
        parts = []
        if "state" in mouth:
            parts.append(f"mouth:{mouth['state']}")
        if "ratio" in mouth:
            parts.append(f"r:{_fmt_float(mouth['ratio'])}")
        if parts:
            lines.append(" ".join(parts))

    if emo:
        parts = []
        if "label" in emo:
            parts.append(f"emo:{emo['label']}")
        if "confidence" in emo:
            parts.append(f"p:{_fmt_float(emo['confidence'])}")
        if parts:
            lines.append(" ".join(parts))

    if "age" in face:
        age = face.get("age")
        adult = face.get("adult")
        lines.append(f"age:{age} adult:{adult}")

    return lines

def draw_json_over_folder(json_path, images_dir, output_dir,
                          person_color=(0,255,0), obj_color=(255,128,0),
                          face_color=(0,255,255)):
    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_drawn = 0
    for frame_id, entries in data.items():
        img_path = _find_image_for_frame(images_dir, frame_id)
        if not img_path:
            print(f"[WARN] No se encontró imagen para {frame_id} en {images_dir}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] No se pudo leer {img_path}")
            continue

        h, w = img.shape[:2]

        for e in entries:
            cls_name = e.get("class_name", "object")
            conf = e.get("confidence", None)
            bbox = e.get("bbox") or e.get("shape")
            if bbox is None:
                continue
            x1,y1,x2,y2 = _ensure_int_bbox(bbox)
            x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue

            if e.get("kind") == "person":
                color = person_color if e.get("kind") == "person" else obj_color
                cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)

                header = f"{cls_name}" + (f" {_fmt_float(conf,3)}" if conf is not None else "")
                _put_multiline(img, (x1, max(18, y1-8)), [header],
                               color=(255,255,255), bg=tuple(int(c*0.6) for c in color), scale=0.5)

            info_lines = []
            if e.get("kind") == "person":
                info_lines += _build_person_lines(e) # pid
            info_lines += _build_depth_lines(e.get("depth")) #profundidad
            if info_lines:
                anchor_y = y2 + 18 if y2 + 60 < h else y1 + 18
                _put_multiline(img, (x1, anchor_y), info_lines, color=(255,255,255), bg=(0,0,0), scale=0.5)

            if e.get("kind") == "person":

                face = (e.get("attributes") or {}).get("face") or {}
                face_bbox = face.get("bbox")
                face_lines = _build_face_panel_lines(face) # datos cara

                if isinstance(face_bbox, (list, tuple)) and len(face_bbox) == 4:
                    fx1, fy1, fx2, fy2 = _ensure_int_bbox(face_bbox)
                    fx1 = max(0, min(fx1, w-1)); fx2 = max(0, min(fx2, w))
                    fy1 = max(0, min(fy1, h-1)); fy2 = max(0, min(fy2, h))
                    if fx2 > fx1 and fy2 > fy1:
                        cv2.rectangle(img, (fx1, fy1), (fx2, fy2), face_color, 2)
                        _put_multiline(img, (fx1, max(18, fy1-8)), ["face"],
                                       color=(0,0,0), bg=face_color, scale=0.5)
                        if face_lines:
                            yy = fy2 + 18 if fy2 + 80 < h else fy1 + 18
                            _put_multiline(img, (fx1, yy), face_lines, color=(255,255,255), bg=(0,0,0), scale=0.5)
                else:
                    if face_lines:
                        anchor_y = y1 + 18 if y1 + 100 < h else max(18, y1 - 8)
                        _put_multiline(img, (min(x2 + 8, w - 10), anchor_y), face_lines,
                                       color=(255,255,255), bg=(0,0,0), scale=0.5)

            total_drawn += 1

        out_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, img)
        print(f"[OK] {frame_id}: anotado → {out_path}")

    print(f"[DONE] Dibujadas {total_drawn} anotaciones en total.")

if __name__ == "__main__":
    draw_json_over_folder(
        json_path=str(paths.RES_DIR / "keyFrames" / "yolo_reid" / "tracking_face_5.json"),
        images_dir=str(paths.KEYFRAMES_SAMPLE_DIR),
        output_dir=str(paths.RES_DIR / "yolo_reid" / "annotated")
    )