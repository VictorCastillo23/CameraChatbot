import cv2
import matplotlib.pyplot as plt
from procesor import extract_keyframes, ForegroundScore, SpatialDifferenceScore, FrequencyDifferenceScore, HistogramDifferenceScore

def Show_table(normalized_scores, keyframes):
    # Mostrar gráfica de puntajes normalizados y keyframes
    plt.figure(figsize=(10, 4))
    plt.plot(normalized_scores, label="Normalized Score")
    plt.scatter(keyframes, [normalized_scores[i] for i in keyframes], color='red', label="Keyframes")
    plt.title("Normalized Scores and Keyframes")
    plt.xlabel("Frame Index")
    plt.ylabel("Normalized Score")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Función para visualizar los resultados de keyframes
def visualize_keyframes(input_video, strategy, scale_percent=20, speed=5, threshold=0.8):
    # Extraer puntuaciones y keyframes usando una estrategia dada
    return extract_keyframes(
        input_video, strategy, scale_percent=scale_percent, speed=speed, threshold=threshold,coments=True
    )

if __name__ == '__main__':
    # Lista de videos a procesar
    video_files = "../videos/input/oficina3.mp4"

    # Estrategia seleccionada (puedes cambiarla por otra)
    strategy = HistogramDifferenceScore()

    # Aplicar visualización a cada video usando map
    normalized_scores, keyframes =  visualize_keyframes(video_files, strategy)

    Show_table(normalized_scores, keyframes)

