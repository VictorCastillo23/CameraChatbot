import cv2, os, numpy as np
from preprocess_vivo import capture_video_chunks
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def ForegroundScore(frame_generator):
    mog2 = cv2.createBackgroundSubtractorMOG2()
    scores = []
    for frame in frame_generator:
        mask = mog2.apply(frame)
        scores.append(round(np.count_nonzero(mask == 255) / frame.size, 4))
    return scores

def normalize_list(values):
    if not values:
        return []
    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        return [0.0 for _ in values]
    return [round((v - min_val) / (max_val - min_val), 4) for v in values]

def find_significant_extrema(signal, prominence=0.05, distance=3, height=None):
    sig = np.asarray(signal, dtype=float)
    peaks, _ = find_peaks(sig, prominence=prominence, distance=distance, height=height)
    return list(peaks)

def overlay_green(frame, alpha=0.4):
    green_layer = np.zeros_like(frame, dtype=np.uint8)
    green_layer[:] = (0, 255, 0)
    return cv2.addWeighted(frame, 1 - alpha, green_layer, alpha, 0)

def extract_keyframes(output_dir="captured_keyframes", fps=30):
    os.makedirs(output_dir, exist_ok=True)

    for i, raw_frames in enumerate(capture_video_chunks(chunk_seconds=5)):
        framesPreprocesaded, normalFrames = raw_frames
        print(f"Clip {i + 1} con {len(framesPreprocesaded)} frames capturados")

        # Calcular score con estrategia
        scores = ForegroundScore(framesPreprocesaded)
        normalized_scores = normalize_list(scores)

        # Detectar keyframes
        keyframes = find_significant_extrema(
            normalized_scores,
            prominence=0.1,
            distance=20,
            height=0.3
        )

        # Crear carpeta del clip
        clip_dir = os.path.join(output_dir, f"clip_{i+1}")
        os.makedirs(clip_dir, exist_ok=True)

        # Configurar escritor de video
        h, w, _ = normalFrames[0].shape
        video_path = os.path.join(clip_dir, f"clip_{i+1}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        for idx, frame in enumerate(normalFrames):
            if idx in keyframes:
                # Guardar frame con filtro verde
                frame_key = overlay_green(frame)
                cv2.imwrite(os.path.join(clip_dir, f"frame_{idx}.jpg"), frame_key)
                for i in range(0,20):
                    out.write(frame_key)
            else:
                out.write(frame)

        out.release()
        print(f"[INFO] Video guardado en: {video_path}")
