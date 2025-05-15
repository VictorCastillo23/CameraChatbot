import cv2
import numpy as np
import time
# modificar doto para que revia un video
class Video:

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

class VideoProcessor:
    def __init__(self, video):
        self.myVideo = Video(video)

    def extract_segment(self, start_time, end_time, output_video_path, frame_skip=1):
        inicio = time.time()
        fps = self.myVideo.get_fps()
        total_frames = self.myVideo.get_frame_count()
        frame_width = self.myVideo.get_width()
        frame_height = self.myVideo.get_height()

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(0, min(end_frame, total_frames - 1))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        self.myVideo.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = self.myVideo.video.read()
            if not ret:
                break
            if current_frame % frame_skip == 0:
                out.write(frame)
            current_frame += 1

        out.release()
        print(f"Segmento guardado en {output_video_path}")
        fin = time.time()
        print(f"El tiempo que tardo en ejecutarse el metodo extract_segment fue de {fin-inicio:.4f} segundos")
        return output_video_path

    def analyze_best_metric(self, sample_size=10):
        inicio = time.time()
        spatial_diff = []
        frequency_diff = []
        histogram_diff = []

        prev_frame = None
        prev_histogram = None

        self.myVideo.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for _ in range(sample_size):
            ret, frame = self.myVideo.video.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            histogram = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
            histogram = cv2.normalize(histogram, histogram).flatten()

            if prev_frame is not None:
                diff = cv2.absdiff(gray_frame, prev_frame)
                spatial_diff.append(np.mean(diff))

                fft_current = np.fft.fft2(gray_frame)
                fft_prev = np.fft.fft2(prev_frame)
                fft_diff = np.abs(fft_current - fft_prev)
                frequency_diff.append(np.mean(fft_diff))

                histogram_dif = cv2.compareHist(histogram, prev_histogram, cv2.HISTCMP_CORREL)
                histogram_diff.append(histogram_dif)

            prev_frame = gray_frame
            prev_histogram = histogram

        variances = {
            "spatial": np.var(spatial_diff),
            "frequency": np.var(frequency_diff),
            "histogram": np.var(histogram_diff)
        }

        best_metric = max(variances, key=variances.get)
        print(f"Métrica óptima para este video: {best_metric}")
        fin = time.time()
        print(f"El tiempo que tardo en ejecutarse el metodo analyze_best_metric fue de {fin - inicio:.4f} segundos")
        return best_metric

    def norm_spatial(self,spatial_diff):
        prev_frame = None
        while self.myVideo.video.isOpened():
            ret, frame = self.myVideo.video.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (64, 64))
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                diff = cv2.absdiff(gray_frame, prev_frame)
                mean_diff = np.mean(diff)
                spatial_diff.append(mean_diff)
            prev_frame = gray_frame
        return min(spatial_diff),max(spatial_diff),spatial_diff

    def norm_frequency(self,frequency_diff):
        prev_frame = None
        while self.myVideo.video.isOpened():
            ret, frame = self.myVideo.video.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (64, 64))
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                fft_current = np.fft.fft2(gray_frame)
                fft_prev = np.fft.fft2(prev_frame)
                fft_diff = np.abs(fft_current - fft_prev)
                mean_fft_diff = np.mean(fft_diff)
                frequency_diff.append(mean_fft_diff)
            prev_frame = gray_frame
        return min(frequency_diff),max(frequency_diff),frequency_diff

    def norm_histogram(self,histogram_diff):
        prev_frame = None
        prev_histogram = None
        while self.myVideo.video.isOpened():
            ret, frame = self.myVideo.video.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (64, 64))
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            histogram = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
            histogram = cv2.normalize(histogram, histogram).flatten()
            if prev_frame is not None:
                histogram_dif = cv2.compareHist(histogram, prev_histogram, cv2.HISTCMP_CORREL)
                histogram_diff.append(histogram_dif)
            prev_frame = gray_frame
            prev_histogram = histogram
        return min(histogram_diff),max(histogram_diff),histogram_diff

    def extract_keyframes2(self, output_video_path, limit_dif_spacial=0.05, limit_dif_FFT=0.1,limit_dif_histogram=0.5,sample_size=10):
        inicio = time.time()
        best_metric = self.analyze_best_metric(sample_size)

        fps = self.myVideo.get_fps()
        width = self.myVideo.get_width()
        height = self.myVideo.get_height()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Listas para almacenar diferencias calculadas
        spatial_diff = []
        frequency_diff = []
        histogram_diff = []

        min_mean_diff = 0
        max_mean_diff = 0
        min_mean_fft_diff = 0
        max_mean_fft_diff = 0
        min_histogram_diff = 0
        max_histogram_diff = 0

        print("Calculando diferencias para normalización")
        self.myVideo.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if best_metric == "spatial":
            min_mean_diff ,max_mean_diff ,spatial_diff = self.norm_spatial(spatial_diff)
            print(f"Normalización calculada: Min/Max Mean Diff: {min_mean_diff}/{max_mean_diff}")
        elif best_metric == "frequency":
            min_mean_fft_diff ,max_mean_fft_diff ,frequency_diff = self.norm_frequency(frequency_diff)
            print(f"Normalización calculada: Min/Max FFT Diff: {min_mean_fft_diff}/{max_mean_fft_diff}")
        else:
            min_histogram_diff ,max_histogram_diff ,histogram_diff = self.norm_histogram(histogram_diff)
            print(f"Normalización calculada:  Min/Max Histogram Diff: {min_histogram_diff}/{max_histogram_diff}")

        fin = time.time()
        print(f"El tiempo que tardo en ejecutarse la normalizacion fue de {fin - inicio:.4f} segundos")
        inicio = time.time()
        print("Detectando fotogramas clave...")
        self.myVideo.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        keyframe_count = 0

        if best_metric == "spatial":
            for i, (mean_diff) in enumerate(spatial_diff):
                normalized_mean_diff = (mean_diff - min_mean_diff) / (max_mean_diff - min_mean_diff)

                print(f"Frame {i}:")
                print(f"  Normalized Mean Diff: {normalized_mean_diff:.2f}")

                if normalized_mean_diff > limit_dif_spacial:
                    print(f"[Cambio rápido] Fotograma clave detectado en {i} con diferencia espacial: {normalized_mean_diff:.2f}")
                    self.myVideo.video.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = self.myVideo.video.read()
                    if ret:
                        out.write(frame)
                        keyframe_count += 1
                frame_count += 1
        elif best_metric == "frequency":
            for i, ( mean_fft_diff) in enumerate( frequency_diff):
                normalized_mean_fft_diff = (mean_fft_diff - min_mean_fft_diff) / (max_mean_fft_diff - min_mean_fft_diff)

                print(f"Frame {i}:")
                print(f"  Normalized FFT Diff: {normalized_mean_fft_diff:.2f}")

                if normalized_mean_fft_diff > limit_dif_FFT:
                    print(
                        f"[Cambio rápido] Fotograma clave detectado en {i} con diferencia frecuencial: {normalized_mean_fft_diff:.2f}")
                    self.myVideo.video.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = self.myVideo.video.read()
                    if ret:
                        out.write(frame)
                        keyframe_count += 1

                frame_count += 1
        else:
            for i, (histogram_diff) in enumerate( histogram_diff):
                normalized_histogram_diff = (histogram_diff - min_histogram_diff) / (max_histogram_diff - min_histogram_diff)

                print(f"Frame {i}:")
                print(f"  Normalized Histogram Diff: {normalized_histogram_diff:.2f}")

                if normalized_histogram_diff < limit_dif_histogram:
                    print(f"[Cambio lento] Fotograma clave detectado en {i} con diferencia de histograma: {normalized_histogram_diff:.2f}")
                    self.myVideo.video.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = self.myVideo.video.read()
                    if ret:
                        out.write(frame)
                        keyframe_count += 1

                frame_count += 1

        out.release()
        print(f"Procesamiento completado. Total de fotogramas clave detectados: {keyframe_count}")
        fin = time.time()
        print(f"El tiempo que tardo en ejecutarse el metodo extract_keyframes fue de {fin- inicio:.4f} segundos")

inicio = time.time()

video_path = ["video_persona.mp4"]
listVideos =[]
listVideos.append(video_path[0])

processor = VideoProcessor(listVideos[0])
video_path.append( processor.extract_segment(0, 65, "segmento.mp4"))
listVideos.append(video_path[1])

processor_seg = VideoProcessor(listVideos[1])
processor_seg.extract_keyframes2("keyframes1.mp4", limit_dif_spacial=0.95, limit_dif_FFT=0.95, limit_dif_histogram=0.2)

fin = time.time()
print(f"El tiempo total que tardo en ejecutarse el codigo fue de {fin- inicio:.4f} segundos")
