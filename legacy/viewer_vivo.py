from keyframe_extractor_vivo import extract_keyframes
import glob, os, matplotlib.pyplot as plt, cv2
def Show_comparison(scores, keyframes, video_name, output_dir="comparisons", behaviorN=1, desired_keyframes=None,name2=''):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 5))

    # Línea de la señal
    plt.plot(scores, color="steelblue", linewidth=2, label="Score")

    # Puntos de los keyframes seleccionados
    if keyframes:
        valid_indices = [idx for idx in keyframes if 0 <= idx < len(scores)]
        y_values = [scores[idx] for idx in valid_indices]
        plt.scatter(
            valid_indices,
            y_values,
            color="crimson",
            s=80,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
            label="Detected Keyframes"
        )

    # 🔹 Líneas verticales para desired keyframes
    if desired_keyframes:
        for dk in desired_keyframes:
            if 0 <= dk < len(scores):
                plt.axvline(x=dk, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
        plt.scatter([], [], color="green", marker="|", s=200, label="Desired Keyframes")  # leyenda

    # Estética para publicación
    #plt.title(f"Keyframes Extraction - scenario_1 - behavior_{behaviorN} - {video_name}", fontsize=22)
    plt.title(f"{name2} - {video_name}", fontsize=22)
    plt.xlabel("frame index", fontsize=20)
    plt.ylabel("Score", fontsize=20)
    plt.grid(alpha=0.3, linestyle="--")
    plt.tick_params(axis="both", labelsize=15)
    plt.legend(fontsize=14)
    plt.tight_layout()

    # Guardar en carpeta
    save_dir = os.path.join(output_dir, f"behavior_{behaviorN}")
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{video_name}_keyframe.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()  # 🔹 Cierra la figura para liberar memoria
    print(f"[INFO] Gráfica guardada en: {output_path}")

def save_keyframes(video_path, keyframes, output_dir="keyframes"):
    # Crear carpeta base si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Crear carpeta de salida específica para el video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(output_dir, video_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir el video: {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for idx in keyframes:
        if 0 <= idx < frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Aquí corregimos la lectura de shape
                h, w, _ = frame.shape   # alto, ancho, canales
                mid = w // 2
                cropped = frame[:, mid:]  # lado derecho

                frame_path = os.path.join(save_dir, f"frame_{idx}.jpg")
                cv2.imwrite(frame_path, cropped)
                print(f"[INFO] Guardado: {frame_path}")
            else:
                print(f"[WARN] No se pudo leer el frame {idx} en {video_name}")

    cap.release()

def visualize_keyframes():
    return extract_keyframes()

if __name__ == '__main__':
    print('\n', ('-' * 40), '\n')
    visualize_keyframes()
    print('\n', ('-' * 40), '\n')
