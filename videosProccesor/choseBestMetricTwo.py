import video_3, pandas as pd, matplotlib.pyplot as plt,operator
import seaborn as sns
import time


def normalizationer(diff, min_mean_diff, max_mean_diff):
    if max_mean_diff == min_mean_diff:
        return [0] * len(diff)
    return [(x - min_mean_diff) / (max_mean_diff - min_mean_diff) for x in diff]

def medir_tiempo(metodo, *args):
    inicio = time.time()
    resultado = metodo(*args)
    fin = time.time()
    return resultado, fin - inicio

video_paths = ['cortomeetraje-niña-parque','cortometraje-robo','movimiento-perro','video_persona','videoclip-arcane']
#video_paths = ["caricatura-conejo"]
#videos = [video_3.Video(f"videos/input/{video_path}.mp4") for video_path in video_paths]

mog2, spatial, frequency, histogram = [], [], [], []

tiempo_spatial, tiempo_frequency, tiempo_histogram, tiempo_mog2 = [], [], [], []
keyframes_spatial, keyframes_frequency, keyframes_histogram, keyframes_mog2 = [], [], [], []

for i, video in enumerate(video_paths):
    analyzer = video_3.VideoAnalyzer(video_3.Video(f"videos/input/{video}.mp4"))

    mog2, t_mog2 = medir_tiempo(analyzer.norm_MOG2, mog2)
    spatial, t_spatial = medir_tiempo(analyzer.norm_spatial, spatial)
    frequency, t_frequency = medir_tiempo(analyzer.norm_frequency, frequency)
    histogram, t_histogram = medir_tiempo(analyzer.norm_histogram, histogram)

    tiempo_mog2.append(t_mog2)
    tiempo_spatial.append(t_spatial)
    tiempo_frequency.append(t_frequency)
    tiempo_histogram.append(t_histogram)

    df = pd.DataFrame({
        "mog2": normalizationer(mog2, min(mog2), max(mog2)),
        "spatial": normalizationer(spatial, min(spatial), max(spatial)),
        "frequency": normalizationer(frequency, min(frequency), max(frequency)),
        "histogram": normalizationer(histogram, min(histogram), max(histogram))
    })

    metricas_df = pd.DataFrame({
        "spatial": spatial,
        "frequency": frequency,
        "histogram": histogram,
        "mog2": mog2
    })

    colors = sns.color_palette("tab10", 4)
    color_mapping = {"mog2": colors[0], "spatial": colors[1], "frequency": colors[2], "histogram": colors[3]}

    sns.heatmap(metricas_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Matriz de Correlación de Métricas")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["mog2"], linestyle="-", label="MOG2", color=color_mapping["mog2"])
    plt.plot(df.index, df["spatial"], linestyle="-", label="Spatial", color=color_mapping["spatial"])
    plt.plot(df.index, df["frequency"], linestyle="-", label="Frequency", color=color_mapping["frequency"])
    plt.plot(df.index, df["histogram"], linestyle="-", label="Histogram", color=color_mapping["histogram"])

    plt.title(f"Comparación de Métricas - {video_paths[i]}")
    plt.xlabel("Índice")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.clf()

    _, n_key_frames = analyzer.extract_keyframes(mog2, min(mog2), max(mog2), 0.75, operator.gt, verbose=False)
    print(f"Keyframes MOG2: {n_key_frames}")

    _, n_key_frames = analyzer.extract_keyframes(spatial, min(spatial), max(spatial), 0.6, operator.gt, verbose=False)
    print(f"Keyframes Spatial: {n_key_frames}")

    _, n_key_frames = analyzer.extract_keyframes(frequency, min(frequency), max(frequency), 0.6, operator.gt,
                                                 verbose=False)
    print(f"Keyframes Frequency: {n_key_frames}")

    _, n_key_frames = analyzer.extract_keyframes(histogram, min(histogram), max(histogram), 0.3, operator.lt,
                                                 verbose=False)
    print(f"Keyframes Histogram: {n_key_frames}")

    print(f"Tiempo Spatial: {tiempo_spatial[i]}")
    print(f"Tiempo Frequency: {tiempo_frequency[i]}")
    print(f"Tiempo Histogram: {tiempo_histogram[i]}")
    print(f"Tiempo MOG2: {tiempo_mog2[i]}")


print(f"Tiempo promedio Spatial: {sum(tiempo_spatial) / len(tiempo_spatial)}")
print(f"Tiempo promedio Frequency: {sum(tiempo_frequency) / len(tiempo_frequency)}")
print(f"Tiempo promedio Histogram: {sum(tiempo_histogram) / len(tiempo_histogram)}")
print(f"Tiempo promedio MOG2: {sum(tiempo_mog2) / len(tiempo_mog2)}")


