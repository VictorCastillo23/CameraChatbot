import json, os,time,re
from datetime import datetime, timezone
from typing import Tuple, Any

from camerachatbot import paths
from camerachatbot.video_schema.timing import frame_timestamp

def reformat_to_video_schema_uniform(
    src_json_path: str,
    dst_json_path: str,
    *,
    video_key: str,
    start_at: str,
    size_xy: Tuple[int, int],
    fps: float,
    per_frame_inference: float,
    per_frame_preprocess: float,
    per_frame_postprocess: float,
) -> str:
    def _parse_start(ts: str) -> datetime:
        if ts.endswith("Z"):
            dt = datetime.fromisoformat(ts[:-1]).replace(tzinfo=timezone.utc)
        else:
            ts_fixed = re.sub(r'(\.\d{1,5})(\+|\-)', lambda m: f"{m.group(1).ljust(7, '0')}{m.group(2)}", ts)

            dt = datetime.fromisoformat(ts_fixed)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
        return dt

    def _iso_z(dt: datetime) -> str:
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def _as_str(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        return f"{v}"

    def _md_scalar(key: str, name: str, dtype: str, value: Any) -> dict:
        return {"metadata_key": key, "name": name, "data_type": dtype, "value": _as_str(value)}

    def _md_object(key: str, name: str, children: list) -> dict:
        return {"metadata_key": key, "name": name, "data_type": "object", "value": None, "children": children}

    def _build_face_metadata(face: dict) -> dict:
        if not isinstance(face, dict):
            return _md_object("face", "face", [])
        children = []
        eyes = face.get("attention")
        if isinstance(eyes, dict):
            eyes_children = []
            if "attention" in eyes: eyes_children.append(_md_scalar("attention", "attention", "string", eyes.get("attention")))
            children.append(_md_object("attention", "attention", eyes_children))
        mouth = face.get("mouth")
        if isinstance(mouth, dict):
            mouth_children = []
            if "state" in mouth: mouth_children.append(_md_scalar("state", "state", "string", mouth.get("state")))
            if "ratio" in mouth: mouth_children.append(_md_scalar("ratio", "ratio", "float",  mouth.get("ratio")))
            children.append(_md_object("mouth", "mouth", mouth_children))
        emotion = face.get("emotion")
        if isinstance(emotion, dict):
            emo_children = []
            if "label" in emotion:      emo_children.append(_md_scalar("label", "label", "string", emotion.get("label")))
            if "confidence" in emotion: emo_children.append(_md_scalar("confidence", "confidence", "float",  emotion.get("confidence")))
            children.append(_md_object("emotion", "emotion", emo_children))
        if "age" in face:
            age_children = []
            if "age_confidence" in face: age_children.append(_md_scalar("confidence", "confidence", "float", face.get("age_confidence")))
            age_children.append(_md_scalar("age", "age", "integer", face.get("age")))
            children.append(_md_object("age", "age", age_children))
        if "adult" in face: children.append(_md_scalar("adult", "adult", "boolean", face.get("adult")))
        return _md_object("face", "face", children)

    def _build_hands_metadata(hands: Any) -> dict:
        children = []
        by_label = {"Right": None, "Left": None}
        if isinstance(hands, list):
            for h in hands:
                if isinstance(h, dict) and h.get("hand") in by_label and by_label[h["hand"]] is None:
                    by_label[h["hand"]] = h
        for label in ("Right", "Left"):
            item = by_label[label]
            if item is None:
                continue
            children.append(_md_object(
                f"hand_{label.lower()}",
                label,
                [_md_scalar("gesture", "gesture", "string", item.get("gesture"))]
            ))
        return _md_object("hands", "hands", children)

    def _entry_to_object(e: dict) -> dict:
        cls = e.get("class_name", "object")
        conf = e.get("confidence")
        bbox = e.get("bbox") or e.get("shape")
        x1=y1=x2=y2=0
        if isinstance(bbox, (list, tuple)) and len(bbox)==4:
            x1,y1,x2,y2 = map(int, bbox)
        elif isinstance(bbox, dict):
            x1=int(bbox.get("x1",0)); y1=int(bbox.get("y1",0))
            x2=int(bbox.get("x2",0)); y2=int(bbox.get("y2",0))
        obj = {
            "class_name": cls,
            "confidence": float(conf) if conf is not None else None,
            "shape": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "metadata": []
        }
        if e.get("kind") == "person":
            if "user_id" in e:
                #print(f'user = {e.get("user_id")}')
                obj["user_id"] = e.get("user_id")
            #if "neighborhood" in e: obj["neighborhood"] = e["neighborhood"]
            attrs = e.get("attributes") or {}
            if "pose" in attrs: obj["metadata"].append(_md_scalar("pose","pose","string",attrs.get("pose")))
            if "pose_conf" in attrs: obj["metadata"].append(_md_scalar("pose_conf","pose_conf","float",attrs.get("pose_conf")))
            obj["metadata"].append(_build_hands_metadata(attrs.get("hands")))
            obj["metadata"].append(_build_face_metadata(attrs.get("face")))
        return obj

    with open(src_json_path, "r", encoding="utf-8") as f:
        src = json.load(f)
    if not isinstance(src, dict):
        raise ValueError("El JSON de entrada debe ser un dict con frames como claves.")

    W, H = int(size_xy[0]), int(size_xy[1])
    t0 = _parse_start(start_at)

    def _sort_key(k: str):
        try:
            return (k)
        except:
            return k

    key_frames = []
    print(f'start_atstart_at = {start_at}')

    for frame_id in sorted(src.keys(), key=_sort_key):

        if "neighborhood" in frame_id:
            continue

        entries = src[frame_id] or []
        objects = []
        for e in entries:
            try:
                objects.append(_entry_to_object(e))
            except Exception as ex:
                print(f"[WARN] frame {frame_id}: entrada ignorada ({ex})")
        try:
            fnum = int(frame_id)
            dt = frame_timestamp(t0, fnum, fps)
            ts = _iso_z(dt)
        except Exception:
            ts = _iso_z(t0)

        #print(f'ts = {ts}')
        kf = {
            "frame_id": str(frame_id),
            "timestamp": ts,
            "size": {"x": W, "y": H},
            "objects": objects,
            "inference": float(per_frame_inference),
            "preprocess": float(per_frame_preprocess),
            "postprocess": float(per_frame_postprocess),
        }

        key_frames.append(kf)

    out = {"video": {"key": video_key,
                     "start_at": _iso_z(t0),
                     "neighborhood" : src["neighborhood"],
                     "key_frames": key_frames}}

    os.makedirs(os.path.dirname(dst_json_path) or ".", exist_ok=True)
    with open(dst_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return dst_json_path

if __name__ == "__main__":
    reformat_to_video_schema_uniform(
        str(paths.RES_DIR / "keyFrames" / "yolo_reid" / "tracking_face_5.json"),
        str(paths.VIDEO_SCHEMA_OUTPUT_PATH),
        video_key='BUCKET_NAME' + 'folder_path',
        start_at=datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        size_xy=(848 , 478),
        fps=30,
        per_frame_inference=1,
        per_frame_preprocess=1,
        per_frame_postprocess=1,
    )