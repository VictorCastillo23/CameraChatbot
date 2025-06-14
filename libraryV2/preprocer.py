import cv2  # Importa OpenCV para procesamiento de video e imágenes

# Función pura que calcula las nuevas dimensiones del frame escalado
def get_scaled_dimensions(width, height, scale_percent):
    # Calcula el nuevo ancho como porcentaje del original
    new_width = int(width * scale_percent / 100)
    # Calcula el nuevo alto como porcentaje del original
    new_height = int(height * scale_percent / 100)
    # Devuelve una tupla con las nuevas dimensiones
    return new_width, new_height

# Función pura que aplica escala, conversión a gris y desenfoque a un frame
def preprocess_frame(frame, width, height, blur_kernel):
    # Convierte el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Redimensiona el frame al tamaño especificado
    resized = cv2.resize(gray, (width, height))
    # Aplica desenfoque gaussiano para reducir ruido
    return cv2.GaussianBlur(resized, blur_kernel, 0)

# Generador que produce frames preprocesados desde un video
def preprocess_video_stream(input_path, scale_percent=10, speed=5, initial_second=0, end_second=0, blur_kernel=(5, 5)):
    # Abre el archivo de video
    video = cv2.VideoCapture(input_path)

    # Verifica si el video se abrió correctamente
    if not video.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {input_path}")

    # Obtiene los frames por segundo del video
    fps = video.get(cv2.CAP_PROP_FPS)
    # Obtiene el número total de frames del video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # Obtiene las dimensiones originales del video
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Calcula las nuevas dimensiones escaladas
    # new_width, new_height = get_scaled_dimensions(width, height, scale_percent)

    # Convierte el segundo inicial a número de frame
    initial_frame = int(fps * initial_second)
    # Convierte el segundo final a frame, o usa el final del video
    end_frame = int(fps * end_second) if end_second > 0 else total_frames

    # Verifica que el rango de tiempo sea válido
    if initial_frame >= end_frame or initial_frame >= total_frames:
        video.release()  # Libera el video antes de lanzar la excepción
        raise ValueError("Rango de tiempo inválido.")

    # Establece el punto inicial de lectura del video
    video.set(cv2.CAP_PROP_POS_FRAMES, initial_frame)

    # Función que determina si se debe procesar el frame actual
    def should_process(current):
        # Procesa solo los frames que cumplen con el salto definido por `speed`
        return (current - initial_frame) % speed == 0

    # Inicializa el frame actual
    current_frame = initial_frame
    # Itera sobre el rango de frames definido
    while current_frame <= end_frame:
        # Lee el siguiente frame del video
        ret, frame = video.read()
        # Si no se pudo leer el frame, termina el bucle
        if not ret:
            break

        # Verifica si este frame debe ser procesado según `speed`
        if should_process(current_frame):
            # Aplica el preprocesamiento al frame y lo entrega
            yield preprocess_frame(frame, 96, 54, blur_kernel)

        # Avanza al siguiente frame
        current_frame += 1

    # Libera el objeto de video después de terminar
    video.release()
