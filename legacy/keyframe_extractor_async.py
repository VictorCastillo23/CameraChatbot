import cv2, os, numpy as np
from scipy.signal import find_peaks

def preprocess_frame(frame, width=96, height=54, blur_kernel=(5, 5)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (width, height))
    return cv2.GaussianBlur(resized, blur_kernel, 0)

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

def extract_keyframes(normalFrames=[],consumer_id=0):

    framesPreprocesaded = [preprocess_frame(f) for f in normalFrames]
    print(f"Clip {consumer_id} con {len(framesPreprocesaded)} frames capturados")

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


    return keyframes