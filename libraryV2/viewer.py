import cv2, os, pandas as pd, matplotlib.pyplot as plt
from procesor2 import extract_keyframes, ForegroundScore, SpatialDifferenceScore, FrequencyDifferenceScore, HistogramDifferenceScore, save_keyframes_as_images
from collections import defaultdict, Counter

def Show_comparison(normalized_scores, keyframes_dict, video_name, strategy_name):
    plt.figure(figsize=(14, 6))
    plt.plot(normalized_scores, label="Normalized Score", color='black', linewidth=1)

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']

    # Alias cortos
    method_aliases = {
        'local_maxima': 'LocalMax',
        'umbral_min_distance': 'Threshold',
        'kmeans': 'KMeans',
        'cumulative_change': 'Cumulative',
        'top_scoring': 'TopK',
        'derivative_change': 'Deriv',
        'union': 'Union',
        'majority_vote': 'Majority',
        'intersection': 'Intersect'
    }

    index_methods = defaultdict(list)
    for method, indices in keyframes_dict.items():
        for idx in indices:
            index_methods[idx].append(method_aliases.get(method, method))

    counts = Counter(sum(keyframes_dict.values(), []))

    for i, (method, indices) in enumerate(keyframes_dict.items()):
        y_values = [normalized_scores[idx] for idx in indices]
        sizes = [60 if counts[idx] > 1 else 40 for idx in indices]
        plt.scatter(indices, y_values, label=method_aliases.get(method, method),
                    color=colors[i % len(colors)], s=sizes, alpha=0.8, edgecolors='k')

    multi_table_data = []
    for idx, methods in sorted(index_methods.items()):
        if len(methods) > 1:
            y = normalized_scores[idx]
            plt.scatter(idx, y, color='gold', s=120, marker='*', edgecolors='black', linewidths=1.2)
            method_str = ", ".join(methods)
            multi_table_data.append((idx, method_str))

    plt.title(f"Comparación de Keyframes - {video_name} usando {strategy_name}")
    plt.xlabel("Índice de Frame")
    plt.ylabel("Score Normalizado")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 📊 Mostrar tabla en una figura aparte si hay coincidencias múltiples
    if multi_table_data and False:
        fig, ax = plt.subplots(figsize=(8, max(2, len(multi_table_data) * 0.4)))
        ax.axis('off')
        col_labels = ["Frame", "Métodos"]
        table_data = [[str(idx), methods] for idx, methods in multi_table_data]
        table = ax.table(cellText=table_data, colLabels=col_labels,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        plt.title("Coincidencias Múltiples de Keyframes")
        plt.tight_layout()
        plt.show()

    if not keyframes_dict:
        print(f"[ADVERTENCIA] No se detectaron keyframes para el video {video_name} con la estrategia {strategy_name}.")
        return

# Función para visualizar los resultados de keyframes
def visualize_keyframes(input_video, strategy, scale_percent=20, speed=5, ):
    # Extraer puntuaciones y keyframes usando una estrategia dada
    return extract_keyframes(
        input_video, strategy, scale_percent=scale_percent, speed=speed,coments=False
    )

if __name__ == '__main__':
    # Estrategia
    strategies = [ForegroundScore()]
    #Lista de videos a procesar
    names_videos = [
        'VID_20250610_205107291~2',
        'VID_20250610_205107291~3',
        'VID_20250610_210055776~2',
        'VID_20250610_210055776~3',
        'VID_20250610_210055776~4',
        'VID_20250611_164125600~2',
        'VID_20250611_164125600~3',
        'VID_20250611_164125600~4',
        'VID_20250611_164125600~5',
        'VID_20250611_164125600~6',
        'VID_20250611_164929985~2',
        'VID_20250611_164929985~3',
        'VID_20250611_164929985~4',
        'VID_20250611_164929985~5',
        'VID_20250611_191323468~2',
        'VID_20250611_191323468~3',
        'VID_20250611_191323468~4',
        'VID_20250611_191323468~5',
        'VID_20250611_191323468~6',
        'VID_20250611_192137805~2',
        'VID_20250611_192137805~3',
        'VID_20250611_192137805~4',
        'VID_20250611_192137805~5',
        'VID_20250611_192137805~6',
        'VID_20250611_192137805~7',
        'VID_20250611_192137805~8',
        'VID_20250611_192903914~2',
        'VID_20250611_192903914~3',
        'VID_20250611_192903914~4',
    ]

    for strategy in strategies:
        for name in names_videos:
            video_files = f"C:/Users/panmo/PycharmProjects/PythonProject/videos/input/{name}.mp4"
            print(f"[Video]  {name}")
            # Aplicar visualización a cada video usando map
            normalized_scores, keyframes =  visualize_keyframes(video_files, strategy)
            print('\n')
            print(keyframes,'\n')
            Show_comparison(normalized_scores, keyframes, video_name = os.path.basename(video_files), strategy_name=strategy.__class__.__name__)

            # Guardar imágenes de los keyframes detectados
            indices_a_guardar = list(keyframes.values())[0] if keyframes else []
            save_keyframes_as_images(
                video_files,
                indices_a_guardar,
                output_dir=f"../videos/keyframes_output/keyframes_{name}",
                scale_percent=20,
                speed=5
            )

        print('\n',('-'*40),'\n')

    #indices_a_guardar = list(keyframes.values())[0] if keyframes else []
    #save_keyframes_as_images(video_files, indices_a_guardar, output_dir=f"keyframes_{name}", scale_percent=20, speed=5)
