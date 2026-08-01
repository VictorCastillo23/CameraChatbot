import cv2, numpy as np,time
from preprocess_mica import preprocess_video_stream
from scipy.signal import find_peaks

def ForegroundScore( frame_generator):
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
    peaks, peak_props = find_peaks(sig, prominence=prominence, distance=distance, height=height)
    return list(peaks)

def extract_keyframes(input_video, speed=1):
    print(f"\n[INFO] Procesando video: {input_video}")
    start_total = time.time()  # tiempo total

    # --- Etapa 1: Preprocesamiento de video ---
    start_pre = time.time()
    raw_frames = list(
        preprocess_video_stream(
            input_path=input_video,
            speed=speed,
            initial_second=0,
            end_second=0,
        )
    )
    end_pre = time.time()
    print(f"[STAGE 1] Preprocesamiento completado ({len(raw_frames)} frames) en {end_pre - start_pre:.2f} segundos")

    # --- Etapa 2: Cálculo de puntajes ---
    start_score = time.time()
    scores = ForegroundScore(raw_frames)
    end_score = time.time()
    print(f"[STAGE 2] Cálculo de scores completado en {end_score - start_score:.2f} segundos")

    # --- Etapa 3: Normalización ---
    start_norm = time.time()
    normalized_scores = normalize_list(scores)
    end_norm = time.time()
    print(f"[STAGE 3] Normalización completada en {end_norm - start_norm:.2f} segundos")

    # --- Etapa 4: Detección de keyframes ---
    start_detect = time.time()
    keyframes = find_significant_extrema(
        normalized_scores,
        prominence=0.05,
        distance=20,
        height=0.3
    )
    end_detect = time.time()
    print(f"[STAGE 4] Detección de keyframes completada en {end_detect - start_detect:.2f} segundos")
    print(f"[RESULT] {len(keyframes)} keyframes detectados")

    # --- Tiempo total ---
    end_total = time.time()
    print(f"[TOTAL] Duración total del proceso: {end_total - start_total:.2f} segundos\n")

    return normalized_scores, keyframes

'''
def extract_keyframes(input_video, speed=1):
    raw_frames = list(
        preprocess_video_stream(input_path=input_video,speed=speed,initial_second=0,end_second=0,)
    )
    scores = ForegroundScore(raw_frames)
    normalized_scores = normalize_list(scores)

    keyframes = find_significant_extrema(normalized_scores,prominence = 0.05,distance= 20,height = 0.3)

    return normalized_scores, keyframes
'''