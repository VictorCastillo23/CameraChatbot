import os,json, cv2,torch,onnxruntime as ort,torchreid
from pathlib import Path
from ultralytics import YOLO
from flask import Flask
from supabase import create_client, Client
from dotenv import load_dotenv

from camerachatbot import paths

load_dotenv()

YOLO_DET_PATH  = (paths.MODELS_DIR / "yolov10m.pt").resolve()
YOLO_POSECLS_PATH = (paths.MODELS_DIR / "trained_yolo11m.pt").resolve()
EMO_ONNX_PATH  = (paths.MODELS_DIR / "emotion-ferplus-8.onnx").resolve()
AGE_ONNX_PATH  = (paths.MODELS_DIR / "age_googlenet.onnx").resolve()
MIDAS_WEIGHTS  = (paths.MODELS_DIR / "dpt_hybrid_384.pt").resolve()

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    _assert_exists(YOLO_DET_PATH, "YOLO detección")
    _assert_exists(YOLO_POSECLS_PATH, "YOLO-CLS pose sentado/de pie")

    yolo_det = YOLO(str(YOLO_DET_PATH))
    yolo_posecls = YOLO(str(YOLO_POSECLS_PATH))

    try:
        yolo_det.fuse()
    except Exception:
        pass
    try:
        yolo_posecls.fuse()
    except Exception:
        pass

    return yolo_det, yolo_posecls

def load_reid(device):
    reid = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=1000,
        pretrained=True
    )
    reid.eval().to(device)

    _, transform = torchreid.data.transforms.build_transforms(height=256, width=128)
    return reid, transform

def load_face_attr_sessions():
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

def load_midas(device):
    _assert_exists(MIDAS_WEIGHTS, "pesos MiDaS DPT_Hybrid")
    depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", pretrained=False)
    state_dict = torch.load(str(MIDAS_WEIGHTS), map_location="cpu")
    depth_model.load_state_dict(state_dict)
    depth_model.eval().to(device)

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    depth_transform = midas_transforms.dpt_transform
    return depth_model, depth_transform

def build_app():
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    return app

def init_runtime():
    device = get_device()
    print(f"[DEVICE] Usando: {device}")

    supabase,BUCKET_NAME = build_supabase()

    app = build_app()

    yolo_det, yolo_posecls = load_yolo_models()
    reid_model, reid_transform = load_reid(device)
    (emotion_sess, emotion_input, emotion_output), (age_sess, age_input, age_output) = load_face_attr_sessions()
    depth_model, depth_transform = load_midas(device)

    RUNTIME = {
        "app": app,
        "supabase": supabase,
        "BUCKET_NAME":BUCKET_NAME,
        "device": device,
        "yolo_det": yolo_det,
        "yolo_posecls": yolo_posecls,
        "reid_model": reid_model,
        "reid_transform": reid_transform,
        "emotion": {
            "sess": emotion_sess,
            "input": emotion_input,
            "output": emotion_output
        },
        "age": {
            "sess": age_sess,
            "input": age_input,
            "output": age_output
        },
        "depth": {
            "model": depth_model,
            "transform": depth_transform
        }
    }
    print("[INIT] Modelos y sesiones cargados correctamente.")
    return RUNTIME
