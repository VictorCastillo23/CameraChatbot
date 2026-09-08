from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = ROOT_DIR / "models"
RES_DIR = ROOT_DIR / "res"
KEYFRAMES_SAMPLE_DIR = ROOT_DIR / "keyFrames"
MIS_FRAMES_DIR = ROOT_DIR / "mis_frames"

GALLERY_INDEX_PATH = ROOT_DIR / "gallery.index"
ID_MAP_PATH = ROOT_DIR / "id_map.json"
PROTO_STORE_PATH = ROOT_DIR / "proto_store.npy"

VIDEO_SCHEMA_OUTPUT_PATH = RES_DIR / "yolo_reid" / "video_schema.json"


def res_output_dir_for(frames_folder: str) -> Path:
    return RES_DIR / os.path.basename(frames_folder)


def debug_boxes_dir_for(frames_folder: str) -> Path:
    return res_output_dir_for(frames_folder) / "yolo_reid" / "box_review"
