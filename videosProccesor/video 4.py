import numpy as np
import cv2
import subprocess
import os
import operator
import time

ffmpeg_path = r"C:\Users\omar_\PycharmProjects\pythonProject\ffmpeg-n6.1-latest-win64-gpl-6.1\bin\ffmpeg.exe"


def load_video(path):
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        raise ValueError(f"No se pudo abrir el archivo de video: {path}")
    return video


def get_video_properties(video):
    return {
        "fps": video.get(cv2.CAP_PROP_FPS),
        "frame_count": int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }


def extract_segment(input_path, output_path, start_time, duration):
    if not os.path.exists(input_path):
        print(f"Error: El archivo de video '{input_path}' no existe.")
        return None

    inicio = time.time()
    command = [
        ffmpeg_path, "-y", "-i", input_path,
        "-ss", str(start_time), "-t", str(duration), "-c", "copy", output_path
    ]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error en FFmpeg:\n{result.stderr}")
            return None
    except Exception as e:
        print(f"Error al ejecutar FFmpeg: {e}")
        return None

    fin = time.time()
    print(f"Segmento guardado en {output_path} en {fin - inicio:.4f} segundos")
    return output_path


def compute_fg_scores(video):
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fg_scores = []
    mog2 = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = video.read()
        if not ret:
            break
        gray = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        fg_mask = mog2.apply(gray)
        fg_scores.append(np.mean(fg_mask))

    return fg_scores


def extract_keyframes(fg_scores, min_lim_frame=150, limit_fg=0.50):
    min_fg, max_fg = min(fg_scores), max(fg_scores)
    normalized_scores = [(score - min_fg) / (max_fg - min_fg) for score in fg_scores]
    keyframes = [i for i, score in enumerate(normalized_scores) if score > limit_fg]
    return [kf for i, kf in enumerate(keyframes) if i == 0 or kf - keyframes[i - 1] >= min_lim_frame]


def create_keyframe_video(video, keyframes, output_path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for kf in keyframes:
        video.set(cv2.CAP_PROP_POS_FRAMES, kf)
        ret, frame = video.read()
        if ret:
            out.write(frame)

    out.release()


def process_video(video_path, output_video_path, limit_fg=0.30):
    video = load_video(video_path)
    props = get_video_properties(video)
    fg_scores = compute_fg_scores(video)
    keyframes = extract_keyframes(fg_scores, limit_fg=limit_fg)
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video position
    create_keyframe_video(video, keyframes, output_video_path, props['fps'], props['width'], props['height'])
    video.release()
    return keyframes


if __name__ == "__main__":
    #   video_paths = ["caricatura-conejo",'cortomeetraje-niña-parque','cortometraje-robo','movimiento-perro','video_persona','videoclip-arcane']

    video_paths = ["cortometraje-robo"]
    process_video(f"videos/input/{video_paths[0]}.mp4",f"videos/output/{video_paths[0]}.mp4")