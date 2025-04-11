import video_3, pandas as pd , matplotlib.pyplot  as plt
import seaborn as sns

def normalizationer(diff, min_mean_diff, max_mean_diff):
    if max_mean_diff == min_mean_diff:
        return [0] * len(diff)
    return [(x - min_mean_diff) / (max_mean_diff - min_mean_diff) for x in diff]


video_paths = ["video_persona", "vid", "vid1", "vic2", "vid3", "vid3.5", "vid4"]
videos = [video_3.Video(f"videos/input/{video_path}.mp4") for video_path in video_paths]

mog2 = []
spatial = []
frequency = []
histogram = []

for i, video in enumerate(video_paths):
    analyzer = video_3.VideoAnalyzer(video_3.Video(f"videos/input/{video}.mp4"))

    mog2 = analyzer.norm_MOG2(mog2)
    spatial = analyzer.norm_spatial(spatial)
    frequency = analyzer.norm_frequency(frequency)
    histogram = analyzer.norm_histogram(histogram)

    df = pd.DataFrame({
        "mog2": normalizationer(mog2,min (mog2),max(mog2)),
        "spatial": normalizationer(spatial,min (spatial),max(spatial)),
        "frequency": normalizationer(frequency,min (frequency),max(frequency)),
        "histogram": normalizationer(histogram,min (histogram),max(histogram))
    })

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["mog2"], linestyle="-", label="MOG2")
    plt.plot(df.index, df["spatial"], linestyle="-", label="Spatial")
    plt.plot(df.index, df["frequency"], linestyle="-", label="Frequency")
    plt.plot(df.index, df["histogram"], linestyle="-", label="Histogram")

    # Personalización
    plt.title(f"Comparación de Métricas - Gráfico de Líneas{video_paths[i]}")
    plt.xlabel("Índice")
    plt.ylabel("Valor")
    plt.legend()  # Mostrar leyenda
    plt.grid(True)  # Agregar cuadrícula

    # Mostrar la gráfica
    plt.show()
