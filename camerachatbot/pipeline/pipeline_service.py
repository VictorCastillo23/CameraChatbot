import os
import time
import json

from camerachatbot import paths
from camerachatbot.pipeline import orchestrator
from camerachatbot.detectors.pose_action_classifier import PoseActionClassifier
from camerachatbot.detectors.face_detector import FaceDetector
from camerachatbot.detectors.face_attributes_detector import FaceAttributesDetector
from camerachatbot.detectors.hand_detector import HandDetector
from camerachatbot.detectors.person_reid import YOLOPersonReID
from camerachatbot.video_schema.formatter import reformat_to_video_schema_uniform
from camerachatbot.db.postgres_writer import json_to_postgre
from camerachatbot.debugging import annotate


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def _safe_div(n, d):
    d = float(d)
    return (float(n) / d) if d > 0 else 0.0


def build_detail_detectors(runtime, frames_folder):
    emotion_sess = runtime["emotion"]["sess"]
    emotion_input = runtime["emotion"]["input"]
    emotion_output = runtime["emotion"]["output"]

    age_sess = runtime["age"]["sess"]
    age_input = runtime["age"]["input"]
    age_output = runtime["age"]["output"]

    return [
        PoseActionClassifier(
            yolo=runtime["yolo_posecls"],
            frames_folder=frames_folder,
            class_map={"sentado": "sentado", "parado": "de pie", "standing": "de pie", "sitting": "sentado"},
            conf_threshold=0.60
        ),
        FaceDetector(frames_folder=frames_folder),
        HandDetector(frames_folder=frames_folder),
        FaceAttributesDetector(
            emotion_input=emotion_input,
            emotion_output=emotion_output,
            age_input=age_input,
            age_output=age_output,
            frames_folder=frames_folder,
            age_sess=age_sess,
            emotion_sess=emotion_sess,
        ),
    ]


def run_pipeline_and_persist(*, runtime, gallery, frames_folder, n_keyframes, size_xy, start_at,
                              video_key, fps=30, final_inference=0.0,
                              draw_debug=False, debug_images_dir=None):
    detail_detectors = build_detail_detectors(runtime, frames_folder)

    output_path = str(paths.res_output_dir_for(frames_folder))
    os.makedirs(output_path, exist_ok=True)
    yolo_output = os.path.join(output_path, "yolo_reid")

    final_json, final_pre_process, final_post_process = orchestrator.multi_models(
        detail_detectors=detail_detectors, keyframes_path=frames_folder,
        gallery=gallery, yoloPersonReID=YOLOPersonReID(
            runtime["yolo_det"], frames_folder, yolo_output,
            runtime["reid_transform"], runtime["reid_model"], runtime["device"],
            depth_model=runtime["depth"]["model"], depth_transform=runtime["depth"]["transform"]
        )
    )
    print(f'final_json = {final_json}')
    print(f'final_pre_process : {final_pre_process}')
    print(f'final_post_process : {final_post_process}')
    total = _safe_float(final_inference) + _safe_float(final_pre_process) + _safe_float(final_post_process)
    print(f'total : {total}')

    pf_inf = _safe_div(final_inference, n_keyframes)
    pf_pre = _safe_div(final_pre_process, n_keyframes)
    pf_post = _safe_div(final_post_process, n_keyframes)

    t0 = time.time()
    reformat_to_video_schema_uniform(
        final_json, str(paths.VIDEO_SCHEMA_OUTPUT_PATH),
        video_key=video_key,
        start_at=start_at,
        size_xy=size_xy,
        fps=fps,
        per_frame_inference=pf_inf,
        per_frame_preprocess=pf_pre,
        per_frame_postprocess=pf_post,
    )
    t1 = time.time()
    print(f"[STAGE POST] reformat_to_video_schema_uniform: {t1 - t0:.3f}s")

    t2 = time.time()
    json_to_postgre(str(paths.VIDEO_SCHEMA_OUTPUT_PATH))
    t3 = time.time()
    print(f"[STAGE POST] json_to_postgre: {t3 - t2:.3f}s")

    try:
        with open(final_json, "r", encoding="utf-8") as f:
            data_out = json.load(f)
        #print(json.dumps(data_out, indent=4, ensure_ascii=False))
    except Exception as e:
        print(f"[WARN] No se pudo imprimir JSON final: {e}")

    if draw_debug:
        annotate.draw_json_over_folder(
            json_path=str(paths.RES_DIR / "yolo_reid" / "tracking_face_5.json"),
            images_dir=debug_images_dir or frames_folder,
            output_dir=str(paths.RES_DIR / "yolo_reid" / "annotated")
        )

    return final_json
