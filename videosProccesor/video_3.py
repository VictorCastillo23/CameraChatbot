import numpy as np, operator, time, random, cv2, subprocess,os
ffmpeg_path = r"C:\Users\omar_\PycharmProjects\pythonProject\ffmpeg-n6.1-latest-win64-gpl-6.1\bin\ffmpeg.exe"

class Video:
    def __init__(self, path):
        self.path = path
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

    def get_path(self):
        return self.path

    def set_frame_position(self, frame):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame)

    def read_frame(self):
        return self.video.read()

class BaseVideoProcessor:
    def __init__(self, video: Video):
        self.video = video

    def process(self):
        raise NotImplementedError("Este método debe ser implementado por las subclases")

class VideoSegmenter():

    def extract_segment(self, input_path, output_path, start_time, duration):
        if not os.path.exists(input_path):
            print(f"Error: El archivo de video '{input_path}' no existe.")
            return None

        inicio = time.time()

        command = [
            ffmpeg_path, "-y",
            "-i", input_path,
            "-ss", str(start_time), "-t", str(duration),
            "-c", "copy", output_path
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

class VideoAnalyzer(BaseVideoProcessor):
    def analyze_best_metric(self, sample_size=100):
        inicio = time.time()
        spatial_diff, frequency_diff, histogram_diff, fg_scores = [], [], [], []
        prev_frame, prev_histogram = None, None

        total_frames = self.video.get_frame_count()
        sample_frames = sorted(random.sample(range(total_frames), min(sample_size, total_frames)))

        mog2 = cv2.createBackgroundSubtractorMOG2()

        for frame_pos in sample_frames:
            self.video.video.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = self.video.video.read()
            if not ret:
                continue

            gray_frame = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY)
            histogram = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
            histogram = cv2.normalize(histogram, histogram).flatten()

            fg_mask = mog2.apply(gray_frame)
            fg_score = np.mean(fg_mask) / 255.0

            if prev_frame is not None:
                spatial_diff.append(np.mean(cv2.absdiff(gray_frame, prev_frame)))
                fft_current, fft_prev = np.fft.fft2(gray_frame), np.fft.fft2(prev_frame)
                frequency_diff.append(np.mean(np.abs(fft_current - fft_prev)))
                histogram_diff.append(cv2.compareHist(histogram, prev_histogram, cv2.HISTCMP_CORREL))
                fg_scores.append(fg_score)

            prev_frame, prev_histogram = gray_frame, histogram

        variances = {
            "spatial": np.var(spatial_diff) if spatial_diff else 0,
            "frequency": np.var(frequency_diff) if frequency_diff else 0,
            "histogram": np.var(histogram_diff) if histogram_diff else 0,
            "fg_score": np.var(fg_scores) if fg_scores else 0
        }

        best_metric = max(variances, key=variances.get)
        fin = time.time()

        print(f"Métrica óptima: {best_metric} (Calculado en {fin - inicio:.4f} segundos)")
        return best_metric

    def norm_spatial(self, spatial_diff):
        self.video.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        prev_frame = None
        while self.video.video.isOpened():
            ret, frame = self.video.read_frame()
            if not ret:
                break
            gray_frame = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY)


            if prev_frame is not None:
                spatial_diff.append(np.mean(cv2.absdiff(gray_frame, prev_frame)))

            prev_frame = gray_frame
        return spatial_diff

    def norm_MOG2(self, fg_scores):
        self.video.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, prev_frame = self.video.read_frame()
        mog2 = cv2.createBackgroundSubtractorMOG2()  # Inicializamos el sustractor de fondo
        while self.video.video.isOpened():
            ret, frame = self.video.read_frame()

            if not ret:
                break
            gray = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reducción de ruido
            fg_mask = mog2.apply(gray)

            fg_score = np.mean(fg_mask)

            if prev_frame is not None:
                fg_scores.append(fg_score)
                #print (fg_score)
            prev_frame=frame
        return fg_scores

    def norm_frequency(self,frequency_diff):
        self.video.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        prev_frame = None
        while self.video.video.isOpened():
            ret, frame = self.video.video.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                fft_current = np.fft.fft2(gray_frame)
                fft_prev = np.fft.fft2(prev_frame)
                fft_diff = np.abs(fft_current - fft_prev)
                frequency_diff.append(np.mean(fft_diff))
            prev_frame = gray_frame
        return frequency_diff

    def norm_histogram(self,histogram_diff,threshold=0.99,min_lim_frame=150):
        self.video.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.video.video.read()
        last_keyframe = -min_lim_frame
        frame_count = 0
        prev_gray = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY) / 255.0
        prev_hist = cv2.calcHist([prev_gray.astype(np.float32)], [0], None, [256], [0, 1])
        prev_hist = cv2.normalize(prev_hist, prev_hist).flatten()

        while self.video.video.isOpened():
            ret, frame = self.video.video.read()
            if not ret:
                break
            gray = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY) / 255.0
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            hist = cv2.calcHist([gray.astype(np.float32)], [0], None, [256], [0, 1])
            hist = cv2.normalize(hist, hist).flatten()
            hist_diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
            histogram_diff.append(hist_diff)
            frame_count+=1
            if (hist_diff > threshold) and (frame_count - last_keyframe >= min_lim_frame):
                prev_hist = hist
                last_keyframe = frame_count


        return histogram_diff

    def extract_keyframes(self, histogram_diff, min_histogram_diff, max_histogram_diff,
                          limit_dif_histogram, op, min_lim_frame=150, verbose=True):
        frame_count = 0
        last_keyframe = -min_lim_frame
        listKeyFrames = []

        for i, diff in enumerate(histogram_diff):
            normalized_histogram_diff = (diff - min_histogram_diff) / (max_histogram_diff - min_histogram_diff)

            if op(normalized_histogram_diff, limit_dif_histogram) and (frame_count - last_keyframe >= min_lim_frame):
                listKeyFrames.append(i)
                if verbose:  # Solo imprime si verbose es True
                    print(f"Frame {i}:")
                    print(f"  Normalized Histogram Diff: {normalized_histogram_diff:.2f}")
                    print(
                        f"[Cambio lento] Fotograma clave detectado en {i} con diferencia de histograma: {normalized_histogram_diff:.2f}")
                last_keyframe = frame_count
                frame_count += 1

        return listKeyFrames, frame_count

    def create_keyframe_video(self, listKeyFrames, out):
        self.video.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        keyframe_count = 0

        for i in listKeyFrames:
            self.video.video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.video.video.read()
            if ret:
                out.write(frame)
                keyframe_count += 1

        return keyframe_count

    def showKeyFrames(self, listKeyFrames,output_video_path,fps,width,height):
        i, j = 0, 0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        while self.video.video.isOpened():
            ret, frame = self.video.read_frame()
            if not ret:
                print("❌ Error: Frame vacío, revisa el video de entrada.")
                break
            #print(f'Frame{i}')
            if j < len(listKeyFrames):
                #print(f'Frame{i}')
                if i == listKeyFrames[j]:
                    j += 1
                    filtered_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    for _ in range(60):

                        out.write(frame)
                else:
                    out.write(frame)
            else:
                out.write(frame)
            i += 1
        out.release()
        print(f"✅ Video procesado correctamente en {output_video_path}.")

    def extract_keyframes2(self, output_video_path, limit_dif_spacial=0.05, limit_dif_FFT=0.1,limit_dif_histogram=0.5,limit_fg=0.50,sample_size=10):
        inicio = time.time()
        best_metric = self.analyze_best_metric(sample_size)

        fps = self.video.get_fps()
        width = self.video.get_width()
        height = self.video.get_height()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        list_diff = []

        print("Calculando diferencias para normalización")
        self.video.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if best_metric == "spatial":
            list_diff = self.norm_spatial(list_diff)
            num_min = min(list_diff)
            num_max = max(list_diff)
            print(f"Normalización calculada: Min/Max Mean Diff: {num_min}/{num_max}")
        elif best_metric == "frequency":
            list_diff = self.norm_frequency(list_diff)
            num_min = min(list_diff)
            num_max = max(list_diff)
            print(f"Normalización calculada: Min/Max FFT Diff: {num_min}/{num_max}")
        elif best_metric == "fg_score":
            list_diff  = self.norm_MOG2(list_diff)
            num_min = min(list_diff)
            num_max = max(list_diff)
            print(f"Normalización calculada: Min/Max fg_score Diff: {num_min}/{num_max}")
        else:
            list_diff = self.norm_histogram(list_diff)
            num_min=min(list_diff)
            num_max=max(list_diff)
            print(f"Normalización calculada:  Min/Max Histogram Diff: {num_min}/{num_max}")

        fin = time.time()
        print(f"El tiempo que tardo en ejecutarse la normalizacion fue de {fin - inicio:.4f} segundos")
        inicio = time.time()
        print("Detectando fotogramas clave...\n")
        self.video.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if best_metric == "spatial":
            listKeyFrames,keyframe_count = self.extract_keyframes(list_diff, num_min, num_max, limit_dif_spacial,operator.gt)
            self.create_keyframe_video(listKeyFrames, out)
        elif best_metric == "frequency":
            listKeyFrames, keyframe_count = self.extract_keyframes(list_diff, num_min, num_max, limit_dif_FFT, operator.gt)
            self.create_keyframe_video(listKeyFrames, out)
        elif best_metric == "fg_score":
            listKeyFrames, keyframe_count = self.extract_keyframes(list_diff, num_min, num_max,limit_fg, operator.gt)
            self.create_keyframe_video(listKeyFrames, out)
        else:
            listKeyFrames, keyframe_count = self.extract_keyframes(list_diff, num_min, num_max,limit_fg, operator.lt)
            self.create_keyframe_video(listKeyFrames, out)

        #self.video.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #newOutput= self.video.get_path().split('.')[0]
        #self.showKeyFrames(listKeyFrames,newOutput+"Full.mp4", fps, width, height)
        out.release()
        print(f"\nProcesamiento completado. Total de fotogramas clave detectados: {keyframe_count}")
        fin = time.time()
        print(f"El tiempo que tardo en ejecutarse el metodo extract_keyframes fue de {fin- inicio:.4f} segundos\n")

if __name__ == "__main__":
    inicio = time.time()
    #   video_paths = ["caricatura-conejo",'cortomeetraje-niña-parque','cortometraje-robo','movimiento-perro','video_persona','videoclip-arcane']

    video_paths = ["caricatura-conejo"]

    videos = [Video(f"videos/input/{video_path}.mp4") for video_path in video_paths]

    segmenter = VideoSegmenter()

    timesVideos = []
    for i, video in enumerate(video_paths):
        inicio1 = time.time()
        segmenter.extract_segment(f"videos/input/{video}.mp4", f"videos/output/segmento{i}.mp4", 0, 40)
        VideoAnalyzer(Video(f"videos/output/segmento{i}.mp4")).extract_keyframes2(f"videos/output/keyframevid{i}.mp4", limit_dif_spacial=0.6, limit_dif_FFT=0.6,limit_dif_histogram=0.3, limit_fg=0.75)
        timesVideos.append(time.time()-inicio1)

    fin = time.time()
    print(f'El tiempo promedio por video fue de {np.mean(timesVideos):.4f} segundos')
    print(f'El tiempo total que tardo en ejecutarse todo el cofigo fue de {fin-inicio:.4f} segundos')

'''
import cv2
import numpy as np

def extract_keyframes(video_path, threshold=0.99, min_frame_gap=150):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el primer frame.")
        return

    prev_gray = cv2.cvtColor(cv2.resize(prev_frame, (64, 64)), cv2.COLOR_BGR2GRAY) / 255.0
    prev_hist = cv2.calcHist([prev_gray.astype(np.float32)], [0], None, [256], [0, 1])
    prev_hist = cv2.normalize(prev_hist, prev_hist).flatten()


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("keyframes4.mp4", fourcc, 30, (prev_frame.shape[1], prev_frame.shape[0]))

    keyframe_count = 0
    frame_count = 0
    last_keyframe = -min_frame_gap  # Para garantizar que el primer frame clave pueda ser guardado

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY) / 255.0
        gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reducción de ruido
        hist = cv2.calcHist([gray.astype(np.float32)], [0], None, [256], [0, 1])
        hist = cv2.normalize(hist, hist).flatten()
        hist_diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)


        # Guardar solo si hay un cambio significativo y ha pasado el mínimo de frames
        if ( hist_diff > threshold) and (frame_count - last_keyframe >= min_frame_gap):
            out.write(frame)
            keyframe_count += 1
            prev_hist = hist
            last_keyframe = frame_count  # Actualizar el último frame clave

        frame_count += 1

    out.release()
    cap.release()
    print(f"Extracción completa: {keyframe_count} fotogramas clave guardados.")

# Uso del script
extract_keyframes('video_persona.mp4')
'''