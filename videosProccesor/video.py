import cv2
import numpy as np
import time
# optical flow
# gradientes
# interfaces con objetos tipo dumi
class VideoProcessor:
    def __init__(self, path):
        self.video = cv2.VideoCapture(path)
        if not self.video.isOpened():
            raise ValueError(f"No se pudo abrir el archivo de video: {path}")

    def get_fps(self):
        return self.video.get(cv2.CAP_PROP_FPS)

    def get_frame_count(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_width(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_height(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def extract_segment(self, start_time, end_time, output_video_path, frame_skip=1):

        fps = self.get_fps()
        total_frames = self.get_frame_count()
        frame_width = self.get_width()
        frame_height = self.get_height()

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(0, min(end_frame, total_frames - 1))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = self.video.read()
            if not ret:
                break
            if current_frame % frame_skip == 0:
                out.write(frame)
            current_frame += 1

        out.release()
        print(f"Segmento guardado en {output_video_path}")
        return output_video_path

    def extract_keyframes(self, output_video_path, limit_dif_spacial=0.05, limit_dif_FFT=0.1,
                          limit_dif_histogram=0.5):
        fps = self.get_fps()
        width = self.get_width()
        height = self.get_height()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Listas para almacenar diferencias calculadas
        spatial_diff = []
        frequency_diff = []
        histogram_diff = []

        prev_frame = None
        prev_histogram = None

        print("Calculando diferencias para normalización")
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            histogram = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
            histogram = cv2.normalize(histogram, histogram).flatten()

            if prev_frame is not None:

                diff = cv2.absdiff(gray_frame, prev_frame)
                
                mean_diff = np.mean(diff)
                spatial_diff.append(mean_diff)

                fft_current = np.fft.fft2(gray_frame)
                fft_prev = np.fft.fft2(prev_frame)
                fft_diff = np.abs(fft_current - fft_prev)
                mean_fft_diff = np.mean(fft_diff)
                frequency_diff.append(mean_fft_diff)

                histogram_dif = cv2.compareHist(histogram, prev_histogram, cv2.HISTCMP_CORREL)
                histogram_diff.append(histogram_dif)

            prev_frame = gray_frame
            prev_histogram = histogram

        min_mean_diff = min(spatial_diff)
        max_mean_diff = max(spatial_diff)
        min_mean_fft_diff = min(frequency_diff)
        max_mean_fft_diff = max(frequency_diff)

        min_histogram_diff = min(histogram_diff)
        max_histogram_diff = max(histogram_diff)

        print(
            f"Normalización calculada: Min/Max Mean Diff: {min_mean_diff}/{max_mean_diff}, Min/Max FFT Diff: {min_mean_fft_diff}/{max_mean_fft_diff}, Min/Max Histogram Diff: {min_histogram_diff}/{max_histogram_diff}")
        print("Detectando fotogramas clave...")
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        keyframe_count = 0

        for i, (mean_diff, mean_fft_diff, histogram_diff) in enumerate(
                zip(spatial_diff, frequency_diff, histogram_diff)):

            normalized_mean_diff = (mean_diff - min_mean_diff) / (max_mean_diff - min_mean_diff)
            normalized_mean_fft_diff = (mean_fft_diff - min_mean_fft_diff) / (max_mean_fft_diff - min_mean_fft_diff)

            normalized_histogram_diff = (histogram_diff - min_histogram_diff) / (
                        max_histogram_diff - min_histogram_diff)

            print(f"Frame {i}:")

            print(f"  Normalized Mean Diff: {normalized_mean_diff:.2f}")
            print(f"  Normalized FFT Diff: {normalized_mean_fft_diff:.2f}")
            print(f"  Normalized Histogram Diff: {normalized_histogram_diff:.2f}")

            if normalized_mean_diff > limit_dif_spacial:
                print(
                    f"[Cambio rápido] Fotograma clave detectado en {i} con diferencia espacial: {normalized_mean_diff:.2f}")
                self.video.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = self.video.read()
                if ret:
                    out.write(frame)
                    keyframe_count += 1
            elif normalized_mean_fft_diff > limit_dif_FFT:
                print(
                    f"[Cambio rápido] Fotograma clave detectado en {i} con diferencia frecuencial: {normalized_mean_fft_diff:.2f}")
                self.video.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = self.video.read()
                if ret:
                    out.write(frame)
                    keyframe_count += 1
            elif normalized_histogram_diff < limit_dif_histogram:
                print(
                    f"[Cambio lento] Fotograma clave detectado en {i} con diferencia de histograma: {normalized_histogram_diff:.2f}")
                self.video.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = self.video.read()
                if ret:
                    out.write(frame)
                    keyframe_count += 1

            frame_count += 1

        out.release()
        print(f"Procesamiento completado. Total de fotogramas clave detectados: {keyframe_count}")
inicio = time.time()

video_path = "video_persona.mp4"

processor = VideoProcessor(video_path)
output_segmento = processor.extract_segment(0, 100, "segmento.mp4")

processor_seg = VideoProcessor(output_segmento)
processor_seg.extract_keyframes("keyframes.mp4", limit_dif_spacial=0.99, limit_dif_FFT=0.99, limit_dif_histogram=0.2)
fin = time.time()
print(f'El tiempo total que tardo en ejecutarse todo el cofigo fue de {fin - inicio:.4f} segundos')
