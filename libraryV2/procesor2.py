import cv2
from preprocer import preprocess_video_stream
import time
import numpy as np
from scipy.signal import argrelextrema
from functools import reduce
from sklearn.cluster import KMeans
from collections import Counter

# Clase base para definir diferentes estrategias de puntuación de frames
class FrameScoringStrategy:
    def score(self, frame_generator):
        raise NotImplementedError("Debes implementar el método 'score'.")  # Metodo abstracto obligatorio

# Estrategia basada en el foreground (MOG2)
class ForegroundScore(FrameScoringStrategy):
    def score(self, frame_generator):
        mog2 = cv2.createBackgroundSubtractorMOG2()  # Inicializa el sustractor de fondo MOG2
        frame_iterator = iter(frame_generator)

        # Descartamos el primer frame
        next(frame_iterator, None)
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

        scores = list(map(compute_score, frame_generator))
        return scores[1:]

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

# Normaliza los valores entre 0 y 1
def normalize_list(values):
    if not values:
        return []
    min_val, max_val = min(values), max(values)
    return [round((v - min_val) / (max_val - min_val), 4) if max_val != min_val else 0.0 for v in values]

import numpy as np
from numpy.polynomial import Polynomial
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from sklearn.metrics import root_mean_squared_error
import warnings

def get_polynomial_keyframes(signal, degree=10, threshold=0.1):
    x_full = np.arange(len(signal))
    y_full = np.array(signal)

    # Buscar primer valor relevante
    start_index = np.argmax(y_full > threshold)

    x_fit = x_full[start_index:]
    y_fit = y_full[start_index:]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        coeffs = np.polyfit(x_fit, y_fit, deg=degree)

    poly = np.poly1d(coeffs)
    d_poly = poly.deriv()
    roots = d_poly.r

    # Filtrar raíces reales dentro del rango útil
    real_roots = [int(round(r.real)) for r in roots if np.isreal(r) and start_index <= r.real < len(signal)]
    keyframes = sorted(set(real_roots))

    return keyframes, poly

def auto_polynomial_keyframes(signal, min_degree=3, max_degree=20,
                              min_roots=2, max_roots=30,
                              alpha=1.0, beta=10.0, plot=False):
    x = np.arange(len(signal))
    best_score = float('inf')
    best_degree = None
    rmse_list = []
    degree_list = []

    for deg in range(min_degree, max_degree + 1):
        keyframes, poly = get_polynomial_keyframes(signal, degree=deg)
        y_fit = poly(x)
        rmse = root_mean_squared_error(signal, y_fit)

        rmse_list.append(rmse)
        degree_list.append(deg)

        # Penaliza si hay muy pocos o demasiados keyframes
        if len(keyframes) < min_roots or len(keyframes) > max_roots:
            penalty = 1.0
        else:
            penalty = 0.0

        score = alpha * rmse + beta * penalty

        if score < best_score:
            best_score = score
            best_degree = deg
            best_keyframes = keyframes
            best_poly = poly

    return best_keyframes, best_degree, best_poly

# Devuelve los índices de los valores que superan un umbral con separación mínima
def keyframes_by_threshold(scores):
    scores = np.array(scores)

    # Escalamiento robusto usando percentiles
    q25, q75 = np.percentile(scores, [25, 75])
    iqr = q75 - q25  # Rango intercuartílico
    upper_bound = q75 + 1.5 * iqr  # límite superior típico para valores "altos"

    # Calculamos un threshold robusto como el máximo entre media + std y este límite superior
    mean = np.mean(scores)
    std = np.std(scores)
    threshold_robust = max(mean + std, upper_bound)

    # Calculamos la proporción de valores altos
    high_score_ratio = np.mean(scores > threshold_robust)

    # Distancia mínima en función de la densidad de valores altos
    if high_score_ratio > 0:
        min_distance = int(len(scores) / (high_score_ratio * len(scores) + 1))  # más separación si hay muchos
    else:
        min_distance = int(0.1 * len(scores))  # valor por defecto si no hay valores altos

    min_distance = max(1, min_distance)

    return indices_above_threshold(scores, threshold_robust, min_distance=min_distance)

def indices_above_threshold(values, threshold, min_distance=None):
    #min_distance = max(1, int(0.05 * len(values))) if min_distance is None else min_distance

    print(f"Threshold: {threshold}\tMin_distance: {min_distance}\t")
    def reducer(acc, current):
        index, value = current
        if value > threshold and (index - acc['last'] >= min_distance):
            acc['indices'].append(index)
            acc['last'] = index
        return acc
    result = reduce(reducer, enumerate(values), {'indices': [], 'last': -min_distance})
    return result['indices']

def keyframes_local_maxima(scores):
    scores = np.array(scores)
    total_len = len(scores)

    # Autoajuste:
    order = max(1, total_len // 100)
    threshold_factor = 0.75 if np.std(scores) > 0.05 else 0.5
    min_distance = max(1, int(0.05 * total_len))

    # Local maxima
    local_maxima = argrelextrema(scores, np.greater, order=order)[0]
    threshold = threshold_factor * np.max(scores)
    strong_maxima = [i for i in local_maxima if scores[i] >= threshold]

    # Filtrado por distancia mínima
    selected = []
    last = -min_distance
    for idx in strong_maxima:
        if idx - last >= min_distance:
            selected.append(idx)
            last = idx
    return selected

'/'
def keyframes_by_cumulative_change(scores):
    scores = np.array(scores)
    total_frames = len(scores)

    # Autoajuste:
    change_threshold = np.mean(np.abs(np.diff(scores))) * 3  # sensible a cambios significativos
    min_distance = max(1, int(0.05 * total_frames))  # más agresivo

    keyframes = []
    last_index = 0
    cumulative_change = 0.0

    for i in range(1, total_frames):
        diff = abs(scores[i] - scores[i - 1])
        cumulative_change += diff

        if (i - last_index >= min_distance) and (cumulative_change >= change_threshold):
            keyframes.append(i)
            last_index = i
            cumulative_change = 0.0

    return keyframes

def keyframes_kmeans(scores):
    scores = np.array(scores)
    total_frames = len(scores)

    # Autoajuste de número de clusters:
    k = min(15, max(3, total_frames // 50))  # flexible según duración del video

    data = np.array([[i, s] for i, s in enumerate(scores)])
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit(data)

    keyframes = []
    for i in range(k):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        cluster_points = data[cluster_indices]
        center = kmeans.cluster_centers_[i]

        distances = np.linalg.norm(cluster_points - center, axis=1)
        closest_index = cluster_indices[np.argmin(distances)]
        keyframes.append(int(data[closest_index][0]))
    return sorted(keyframes)

def keyframes_by_derivative_change_adaptive(scores):
    scores = np.array(scores)
    derivative = np.abs(np.diff(scores))
    N = len(scores)

    # Umbral dinámico: percentil 85 del gradiente (no absoluto)
    threshold = np.percentile(derivative, 85)

    # Distancia mínima adaptativa: 3% a 10% del total de frames
    min_distance_ratio = min(0.1, max(0.05, 30 / N))  # al menos 30 frames de separación si posible
    min_distance = max(1, int(min_distance_ratio * N))

    selected = []
    last = -min_distance
    for i, val in enumerate(derivative):
        if val >= threshold and (i - last >= min_distance):
            selected.append(i + 1)  # i+1 porque np.diff reduce el tamaño en 1
            last = i
    return selected

def top_scoring_with_suppression_adaptive(scores):
    scores = np.array(scores)
    N = len(scores)

    # Ajuste automático:
    top_k = max(3, min(15, N // 30))  # Entre 3 y 15 keyframes
    min_distance = max(1, int(0.05 * N))  # 5% de distancia entre keyframes

    indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    selected = []
    for idx, score in indexed_scores:
        if all(abs(idx - s) >= min_distance for s in selected):
            selected.append(idx)
        if len(selected) >= top_k:
            break
    return sorted(selected)

def combined_keyframe_selector(scores):
    scores = np.array(scores)
    normalized = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)

    local = keyframes_local_maxima(scores)
    umbral = keyframes_by_threshold(scores)
    kmeans = keyframes_kmeans(scores)
    cumulative = keyframes_by_cumulative_change(normalized)
    suppression = top_scoring_with_suppression_adaptive(scores)
    derivative = keyframes_by_derivative_change_adaptive(scores)

    all_frames = local + umbral + kmeans + cumulative + suppression + derivative
    counts = Counter(all_frames)

    union = sorted(set(all_frames))
    intersection = sorted([frame for frame, count in counts.items() if count >= 4])
    majority_vote = sorted([frame for frame, count in counts.items() if count >= 3])

    #print(f"Local Maxima Keyframes ({len(local)}):\n{local}\n")
    print(f"Threshold Keyframes ({len(umbral)}):\n{umbral}\n")
    #print(f"KMeans Keyframes ({len(kmeans)}):\n{kmeans}\n")
    #print(f"Cumulative Change Keyframes ({len(cumulative)}):\n{cumulative}\n")
    #print(f"Top Scoring Suppression Keyframes ({len(suppression)}):\n{suppression}\n")
    #print(f"Derivative Change Keyframes ({len(derivative)}):\n{derivative}\n")

    #print(f"Union of All Keyframes ({len(union)}):\n{union}\n")
    #print(f"Intersection (Count >= 4) Keyframes ({len(intersection)}):\n{intersection}\n")
    #print(f"Majority Vote (Count >= 3) Keyframes ({len(majority_vote)}):\n{majority_vote}\n")
    return {
        'local_maxima': local,
        'umbral_min_distance': umbral,
        'kmeans': kmeans,
        'cumulative_change': cumulative,
        'top_scoring_suppression': suppression,
        'derivative_change': derivative,
        'union': union,
        'majority_vote': majority_vote,
        'intersection': intersection
    }
# Función principal para procesamiento, acepta una estrategia de puntuación
def extract_keyframes(input_video, scoring_strategy, scale_percent=50, speed=5,coments = False):
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
    keyframes, degree, poly = auto_polynomial_keyframes(np.array(normalized_scores))
    print('Original list',normalized_scores)
    print("Grado óptimo:", degree)
    print("Keyframes:", keyframes)
    keyframe_list = keyframes  # Guarda la lista original

    keyframes = {}  # Ahora sí puedes sobrescribir
    keyframes['polinomio'] = keyframe_list

   # results = combined_keyframe_selector(normalized_scores)
   # keyframes['local_maxima'] = results['local_maxima']
   # keyframes['umbral_min_distance'] = results['umbral_min_distance']
   # keyframes['kmeans'] = results['kmeans']
   # keyframes['cumulative_change'] = results['cumulative_change']
   # keyframes['top_scoring_suppression'] = results['top_scoring_suppression']
   # keyframes['derivative_change'] = results['derivative_change']

    #keyframes['majority_vote'] = results['majority_vote']
    #keyframes['intersection'] = results['intersection']
    if coments:
        print(f"[Tiempo] Selección de keyframes: {time.time() - start:.3f} segundos")

    return normalized_scores, keyframes

def save_keyframes_as_images(input_video, keyframes_indices, output_dir="keyframes_images", scale_percent=20, speed=1):
    import os
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(input_video)
    index = 0
    saved = 0
    keyframe_set = set(keyframes_indices)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #print(f"[DEBUG] Leyendo frame {index}")

        if index % speed == 0:
            current_frame_index = index // speed
            #print(f"[DEBUG] Frame filtrado (cada {speed}): {current_frame_index}")

            if current_frame_index in keyframe_set:
                #print(f"[DEBUG] Guardando frame {current_frame_index}")
                resized = cv2.resize(frame, (
                    int(frame.shape[1] * scale_percent / 100),
                    int(frame.shape[0] * scale_percent / 100)
                ))
                path = os.path.join(output_dir, f"frame_{current_frame_index:05}.jpg")
                cv2.imwrite(path, resized)
                saved += 1

        index += 1

    cap.release()