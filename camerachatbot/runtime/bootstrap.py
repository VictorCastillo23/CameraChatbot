import os, json, cv2, onnxruntime as ort
from pathlib import Path
from flask import Flask
from supabase import create_client, Client
from dotenv import load_dotenv

from camerachatbot import paths
from camerachatbot.security_config import DETECTOR_FLAGS

# Fase 3c: the torch/ultralytics/torchreid runtime path (`loaders_torch.py`,
# the isinstance-based duck typing this used to need in person_reid.py/
# pose_action_classifier.py, and the matching requirements.txt pins) was
# removed entirely — `loaders_onnx` is now the only loader module. There is
# no more one-line rollback; reverting to the torch path would mean
# reverting to a pre-Fase-3c git commit, not flipping an import line.
from camerachatbot.runtime.loaders_onnx import load_detector, load_posecls, load_reid

load_dotenv()

EMO_ONNX_PATH  = (paths.MODELS_DIR / "emotion-ferplus-8.onnx").resolve()
AGE_ONNX_PATH  = (paths.MODELS_DIR / "age_googlenet.onnx").resolve()

def _assert_exists(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(f"[CONFIG] No se encontró {what}: {path}")

def onnx_providers():
    av = ort.get_available_providers()
    if "CUDAExecutionProvider" in av:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

def build_supabase():
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    print(f'SUPABASE_URL: {SUPABASE_URL}')
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    print(f'SUPABASE_KEY: {SUPABASE_KEY}')

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("[WARN] SUPABASE_URL / SUPABASE_KEY no configurados en .env")
        return None,None
    try:
        client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        return client, os.getenv("SUPABASE_BUCKET")
    except Exception as e:
        print(f"[WARN] Supabase init falló: {e}")
        return None,None

def load_yolo_models():
    """Load the person/object detector + sit/stand pose classifier.

    Delegates to `loaders_onnx` (the only loader module since Fase 3c) —
    this function no longer contains any model-loading logic of its own, it
    is purely the two-value-tuple adapter `init_runtime()` expects.
    """
    yolo_det = load_detector()
    yolo_posecls = load_posecls()
    return yolo_det, yolo_posecls

def load_face_attr_sessions():
    if not (DETECTOR_FLAGS["emotion"] or DETECTOR_FLAGS["age"]):
        print("[ONNX] DETECTOR_FLAGS['emotion'] y ['age'] son False — no se cargan sesiones ONNX de emoción/edad.")
        return (None, None, None), (None, None, None)

    _assert_exists(EMO_ONNX_PATH, "modelo de emociones ONNX")
    _assert_exists(AGE_ONNX_PATH, "modelo de edad ONNX")

    providers = onnx_providers()
    print(f"[ONNX] Providers: {providers}")

    emotion_sess = ort.InferenceSession(str(EMO_ONNX_PATH), providers=providers)
    age_sess     = ort.InferenceSession(str(AGE_ONNX_PATH), providers=providers)

    emo_in  = emotion_sess.get_inputs()[0].name
    emo_out = emotion_sess.get_outputs()[0].name
    age_in  = age_sess.get_inputs()[0].name
    age_out = age_sess.get_outputs()[0].name

    print("[EMO] input shape:", emotion_sess.get_inputs()[0].shape)
    print("[AGE] input shape:", age_sess.get_inputs()[0].shape)

    return (emotion_sess, emo_in, emo_out), (age_sess, age_in, age_out)

def build_app():
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    return app

def init_runtime():
    supabase,BUCKET_NAME = build_supabase()

    app = build_app()

    yolo_det, yolo_posecls = load_yolo_models()

    # `loaders_onnx.load_reid()` returns a single self-contained
    # `OSNetOnnxEmbedder` that owns its own preprocessing — since Fase 3c
    # there is no other loader module and no other shape to absorb here.
    reid_model = load_reid()

    (emotion_sess, emotion_input, emotion_output), (age_sess, age_input, age_output) = load_face_attr_sessions()

    RUNTIME = {
        "app": app,
        "supabase": supabase,
        "BUCKET_NAME":BUCKET_NAME,
        "yolo_det": yolo_det,
        "yolo_posecls": yolo_posecls,
        "reid_model": reid_model,
        "emotion": {
            "sess": emotion_sess,
            "input": emotion_input,
            "output": emotion_output
        },
        "age": {
            "sess": age_sess,
            "input": age_input,
            "output": age_output
        }
    }
    print("[INIT] Modelos y sesiones cargados correctamente.")
    return RUNTIME
