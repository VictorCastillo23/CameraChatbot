import cv2  # Importa la librería OpenCV para procesamiento de imágenes
import numpy as np  # Importa NumPy para operaciones numéricas
from preprocer import preprocess_video_stream  # Importa la función de preprocesamiento de video
from functools import reduce  # Importa reduce para operaciones funcionales
import time

# Clase base para definir diferentes estrategias de puntuación de frames
class FrameScoringStrategy:
    def score(self, frame_generator):
        raise NotImplementedError("Debes implementar el método 'score'.")  # Metodo abstracto obligatorio

# Estrategia basada en el foreground (MOG2)
class ForegroundScore(FrameScoringStrategy):
    def score(self, frame_generator):
        mog2 = cv2.createBackgroundSubtractorMOG2()  # Inicializa el sustractor de fondo MOG2
        return [
            round(np.count_nonzero(mog2.apply(frame) == 255) / frame.size, 4)
            for frame in frame_generator
        ]

# Estrategia basada en diferencias de histogramas
class HistogramDifferenceScore(FrameScoringStrategy):
    def score(self, frame_generator):
        prev_hist = None

        def compute_score(frame):
            nonlocal prev_hist
            norm_frame = frame / 255.0
            hist = cv2.calcHist([norm_frame.astype(np.float32)], [0], None, [256], [0, 1])
            hist = cv2.normalize(hist, hist).flatten()
            if prev_hist is not None:
                score = round(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR), 4)
            else:
                score = 0.0
            prev_hist = hist
            return score

        return list(map(compute_score, frame_generator))

# Estrategia basada en diferencias espaciales (frame a frame)
class SpatialDifferenceScore(FrameScoringStrategy):
    def score(self, frame_generator):
        prev_frame = None
        def compute_score(frame):
            nonlocal prev_frame
            if prev_frame is not None:
                score = round(np.mean(cv2.absdiff(frame, prev_frame)), 4)
            else:
                score = 0.0
            prev_frame = frame
            return score

        return list(map(compute_score, frame_generator))

# Estrategia basada en diferencias de frecuencia (FFT)
class FrequencyDifferenceScore(FrameScoringStrategy):
    def score(self, frame_generator):
        prev_frame = None

        def compute_score(frame):
            nonlocal prev_frame
            if prev_frame is not None:
                fft_current = np.fft.fft2(frame)
                fft_prev = np.fft.fft2(prev_frame)
                fft_diff = np.abs(fft_current - fft_prev)
                score = round(np.mean(fft_diff), 4)
            else:
                score = 0.0
            prev_frame = frame
            return score

        return list(map(compute_score, frame_generator))

# Normaliza los valores entre 0 y 1 con 4 decimales
def normalize_list(values):
    if not values:
        return []
    min_val, max_val = min(values), max(values)
    return [round((v - min_val) / (max_val - min_val), 4) if max_val != min_val else 0.0 for v in values]

# Devuelve los índices de los valores que superan un umbral con separación mínima
def indices_above_threshold(values, threshold, min_distance=None):
    min_distance = max(1, int(0.05 * len(values))) if min_distance is None else min_distance

    def reducer(acc, current):
        index, value = current
        if value > threshold and (index - acc['last'] >= min_distance):
            acc['indices'].append(index)
            acc['last'] = index
        return acc

    result = reduce(reducer, enumerate(values), {'indices': [], 'last': -min_distance})
    return result['indices']

# Función principal para procesamiento, acepta una estrategia de puntuación

def extract_keyframes(input_video, scoring_strategy, scale_percent=50, speed=5, threshold=0.5,coments = False):
    # Medir tiempo del preprocesamiento
    start = time.time()
    frames = preprocess_video_stream(
        input_path=input_video,
        scale_percent=scale_percent,
        speed=speed,
        initial_second=0,
        end_second=0
    )
    frames = list(frames)  # Importante: convertir a lista para reutilizar el generador
    preprocess_time = time.time() - start
    if coments:
        print(f"[Tiempo] Preprocesamiento: {preprocess_time:.3f} segundos")

    # Medir tiempo del scoring
    start = time.time()
    scores = scoring_strategy.score(frames)
    if coments:
        print(f"[Tiempo] Scoring ({scoring_strategy.__class__.__name__}): {time.time() - start:.3f} segundos")

    # Medir tiempo de normalización
    start = time.time()
    normalized_scores = normalize_list(scores)
    if coments:
        print(f"[Tiempo] Normalización: {time.time() - start:.3f} segundos")

    # Medir tiempo de selección de keyframes
    start = time.time()
    keyframes = indices_above_threshold(normalized_scores, threshold)
    if coments:

        print(f"[Tiempo] Selección de keyframes: {time.time() - start:.3f} segundos")

    return normalized_scores, keyframes

