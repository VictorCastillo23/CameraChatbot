import cv2, time

def preprocess_frame(frame, width=96, height=54, blur_kernel=(5, 5)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (width, height))
    return cv2.GaussianBlur(resized, blur_kernel, 0)

def capture_video_chunks(chunk_seconds=5):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo acceder a la cámara.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    frames_per_chunk = int(fps * chunk_seconds)

    try:
        while True:
            framesPreprocesaded  = []
            normalFrames  = []
            start_time = time.time()

            for _ in range(frames_per_chunk):
                ret, frame = cap.read()
                if not ret:
                    break
                normalFrames.append(frame)
                framesPreprocesaded.append(preprocess_frame(frame))

            if framesPreprocesaded:
                yield framesPreprocesaded, normalFrames

            elapsed = time.time() - start_time
            if elapsed < chunk_seconds:
                time.sleep(chunk_seconds - elapsed)

    finally:
        cap.release()
        cv2.destroyAllWindows()
