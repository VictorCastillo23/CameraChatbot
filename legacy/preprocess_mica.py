import cv2

def preprocess_frame(frame, width, height, blur_kernel):
    # Convertir a gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mid = w // 2
    cropped = gray[:, mid:]
    # Redimensionar
    resized = cv2.resize(gray, (width, height))
    # Suavizado
    return cv2.GaussianBlur(resized, blur_kernel, 0)

def preprocess_video_stream(input_path, speed=1, initial_second=0, end_second=0, blur_kernel=(5, 5)):
    video = cv2.VideoCapture(input_path)
    if not video.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {input_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    initial_frame = int(fps * initial_second)
    end_frame = int(fps * end_second) if end_second > 0 else total_frames

    if initial_frame >= end_frame or initial_frame >= total_frames:
        video.release()
        raise ValueError("Rango de tiempo inválido.")

    video.set(cv2.CAP_PROP_POS_FRAMES, initial_frame)

    def should_process(current):
        return (current - initial_frame) % speed == 0

    current_frame = initial_frame

    while current_frame <= end_frame:
        ret, frame = video.read()
        if not ret:
            break
        if should_process(current_frame):
            yield preprocess_frame(frame, 96, 54, blur_kernel)
        current_frame += 1

    video.release()
    cv2.destroyAllWindows()