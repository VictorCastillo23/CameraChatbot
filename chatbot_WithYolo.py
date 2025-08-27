from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
import json
import mysql.connector
from datetime import datetime
from dotenv import load_dotenv
from collections import defaultdict
import glob
import time
import matplotlib.pyplot as plt
import psutil

load_dotenv()

class Chatbot_WITH_YOLO():
    def __init__(self, mysql_config):
        self.api_key = self.load_api_key()
        os.environ['OPENAI_API_KEY'] = self.api_key
        self.llm = self.initialize_llm()
        self.mysql_config = mysql_config
        self.init_db()

    def load_api_key(self):
        return os.getenv('OPENAI_API_KEY')

    def initialize_llm(self):
        return ChatOpenAI(model="gpt-4o", temperature=0)

    def init_db(self):
        conn = mysql.connector.connect(**self.mysql_config)
        cursor = conn.cursor()

        # Crear tabla para historias por imagen
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stories (
                id INT AUTO_INCREMENT PRIMARY KEY,
                carpeta VARCHAR(255),
                imagen VARCHAR(255),
                general_description TEXT,
                objects_list TEXT,
                collision_analysis TEXT,
                narrative TEXT,
                created_at DATETIME
            )
        ''')

        # Crear tabla para narrativa general por video
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stories_globales (
                id INT AUTO_INCREMENT PRIMARY KEY,
                carpeta VARCHAR(255),
                narrativa_general TEXT,
                created_at DATETIME
            )
        ''')

        conn.commit()
        cursor.close()
        conn.close()

    def guardar_story_bd(self, carpeta, imagen, general_desc, objects_list, collision_analysis, narrative):
        conn = mysql.connector.connect(**self.mysql_config)
        cursor = conn.cursor()
        sql = """
        INSERT INTO stories 
            (carpeta, imagen, general_description, objects_list, collision_analysis, narrative, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        val = (carpeta, imagen, general_desc, objects_list, collision_analysis, narrative, datetime.now())
        cursor.execute(sql, val)
        conn.commit()
        cursor.close()
        conn.close()

    def guardar_narrativa_general(self, carpeta, narrativa):
        conn = mysql.connector.connect(**self.mysql_config)
        cursor = conn.cursor()
        sql = """
        INSERT INTO stories_globales (carpeta, narrativa_general, created_at)
        VALUES (%s, %s, %s)
        """
        val = (carpeta, narrativa, datetime.now())
        cursor.execute(sql, val)
        conn.commit()
        cursor.close()
        conn.close()

    def analizar_conclusiones_y_guardar(self, keyframes_root: str) -> dict:
        start_global = time.time()

        narrativas_por_carpeta = defaultdict(list)
        resultados = {}
        tiempos_por_video = {}
        memoria_por_video = {}
        total_respuestas = 0
        respuestas_validas = 0

        # Recorrer todas las imágenes en todos los subdirectorios de manera recursiva
        imagenes = glob.glob(os.path.join(keyframes_root, "**", "*.jpg"), recursive=True)

        for imagen_path in imagenes:
            carpeta_relativa = os.path.relpath(os.path.dirname(imagen_path), keyframes_root)
            image_name = os.path.basename(imagen_path)

            start_video = time.time()
            process = psutil.Process(os.getpid())

            prompt = (
                f"You will receive an image with border boxes for object detection, corresponding to a keyframe from a video. "
                f"Analyze the image '{image_name}' in the folder '{carpeta_relativa}', coming from a SECURITY CAMERA IN A CAR. "
                "Describe what is happening in the scene.\n\n"
                "Return ONLY a JSON with the following keys:\n"
                "- general_description: a general description of what is observed in the image.\n"
                "- objects_list: if possible, a list of visible objects (even if no specific data is available).\n"
                "- collision_analysis: if there is visible movement or interaction, interpret possible collisions or relationships.\n"
                "- narrative: a free and creative narrative of the scene, as if telling a story.\n\n"
                "Please respond ONLY with plain JSON, without ```json or any other formatting."
            )

            msg = HumanMessage(content=prompt)
            response = self.llm.invoke([msg])
            response_text = response.content.strip()
            total_respuestas += 1

            try:
                parsed = json.loads(response_text)
                respuestas_validas += 1
                general_desc = parsed.get("general_description", "")
                objects_list = parsed.get("objects_list", "")
                if isinstance(objects_list, list):
                    objects_list = "\n".join(f"- {obj}" for obj in objects_list)
                collision_analysis = parsed.get("collision_analysis", "")
                narrative = parsed.get("narrative", "")

                if narrative:
                    narrativas_por_carpeta[carpeta_relativa].append(narrative)

                resultados[f"{carpeta_relativa}/{image_name}"] = parsed

                self.guardar_story_bd(
                    carpeta_relativa,
                    image_name,
                    general_desc,
                    objects_list,
                    collision_analysis,
                    narrative
                )
            except json.JSONDecodeError:
                resultados[f"{carpeta_relativa}/{image_name}"] = {
                    "error": "Could not parse response as JSON.",
                    "raw_response": response_text
                }
                self.guardar_story_bd(
                    carpeta_relativa,
                    image_name,
                    "",
                    "",
                    "",
                    response_text
                )

            print(f"Imagen {carpeta_relativa}/{image_name} procesada.")

            # Guardar memoria y tiempo por carpeta relativa
            memoria_actual = process.memory_info().rss / (1024 ** 2)
            memoria_por_video[carpeta_relativa] = max(memoria_por_video.get(carpeta_relativa, 0), memoria_actual)
            tiempos_por_video[carpeta_relativa] = tiempos_por_video.get(carpeta_relativa, 0) + (
                        time.time() - start_video)

        # Generar narrativa global por carpeta/video
        for carpeta, narrativas in narrativas_por_carpeta.items():
            resumen_prompt = (
                    f"You have received a series of narratives corresponding to images extracted from a SECURITY CAMERA IN A CAR. "
                    f"Based on these narratives, generate a coherent overall narrative of the video.\n\n"
                    f"List of narratives per image:\n\n"
                    + "\n\n".join(f"- {n}" for n in narrativas)
                    + "\n\nReturn ONLY one overall narrative in plain text, without JSON formatting."
            )

            msg = HumanMessage(content=resumen_prompt)
            response = self.llm.invoke([msg])
            narrativa_general = response.content.strip()

            print(f"\nNarrativa general para el video (carpeta: {carpeta}):\n{narrativa_general}\n")
            self.guardar_narrativa_general(carpeta, narrativa_general)

        print(f"\n✅ (WITH YOLO) Tiempo total: {time.time() - start_global:.2f} segundos.")

        # Graficar tiempos por video
        videos = list(tiempos_por_video.keys())
        tiempos = list(tiempos_por_video.values())

        # Crear etiquetas numeradas para los videos
        videos_numerados = [f"Video {i + 1}" for i in range(len(videos))]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(videos_numerados, tiempos, color='skyblue')

        plt.title("Processing time per video (WITH YOLO)", fontsize=18)
        plt.xlabel("Video (folder)", fontsize=14)
        plt.ylabel("Time (seconds)", fontsize=14)
        plt.xticks(rotation=90, ha='center')

        for bar, tiempo in zip(bars, tiempos):
            altura = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                altura + 0.5,
                f"{tiempo:.2f}s",
                ha='center',
                va='bottom',
                rotation=90
            )

        plt.tight_layout()
        plt.savefig("tiempos_por_video_WITH_YOLO.png")
        plt.show()

        # Gráfica de uso de memoria por video
        videos_memoria = list(memoria_por_video.keys())
        usos_memoria = list(memoria_por_video.values())
        # Crear etiquetas numeradas para los videos
        videos_numerados = [f"Video {i + 1}" for i in range(len(videos_memoria))]

        plt.figure(figsize=(10, 6))
        bars_memoria = plt.bar(videos_numerados, usos_memoria, color='lightcoral')

        plt.title("Memory usage per video (WITH YOLO)", fontsize=18)
        plt.xlabel("Video (folder)", fontsize=14)
        plt.ylabel("Peak memory (MB)", fontsize=14)
        plt.xticks(rotation=90, ha='center')

        # Etiquetas dentro de cada barra (verticales y negras)
        for bar, mem in zip(bars_memoria, usos_memoria):
            altura = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                altura - 5,  # dentro de la barra
                f"{mem:.2f} MB",
                ha='center',
                va='top',
                rotation=90,
                color='black',
                fontsize=9,
                fontweight='bold'
            )

        plt.tight_layout()
        plt.savefig("memoria_por_video_WITH_YOLO.png")
        plt.show()

        # Promedio de uso de memoria (sólo al final)
        self.promedio_memoria_mb = sum(memoria_por_video.values()) / len(memoria_por_video) if memoria_por_video else 0

        precision_global = (respuestas_validas / total_respuestas) * 100 if total_respuestas > 0 else 0
        print(
            f"\n🎯 Precisión de YOLO parseado correctamente: {precision_global:.2f}% ({respuestas_validas}/{total_respuestas})")

        precision_por_video = {video: precision_global for video in videos_memoria}

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            [tiempos_por_video[v] for v in videos_memoria],
            [memoria_por_video[v] for v in videos_memoria],
            c=[precision_por_video[v] for v in videos_memoria],
            cmap='viridis', s=100, alpha=0.7
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label('Accuracy (%)', fontsize=14)  # aumenta solo el label de la barra
        cbar.ax.tick_params(labelsize=10)  # tamaño de etiquetas en la barra
        plt.title("Trade-off between Latency, Memory and Accuracy (WITH YOLO)", fontsize=18)
        plt.xlabel("Time (s)", fontsize=14)
        plt.ylabel("Memory (MB)", fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("compensacion_latencia_memoria_precision_WITH_YOLO.png")
        plt.show()

        return resultados

    def execute_chatbot_withYolo(self):
        mysql_config = {
            'user': os.getenv('MYSQL_USER'),
            'password': os.getenv('MYSQL_PASSWORD'),
            'host': os.getenv('MYSQL_HOST'),
            'database': os.getenv('MYSQL_DATABASE'),
            'port': int(os.getenv('MYSQL_PORT', 3306)),
        }

        chatbot = Chatbot_WITH_YOLO(mysql_config=mysql_config)

        keyframes_root = r"C:/Users/panmo/PycharmProjects/PythonProject/YOLO/runs/resultados_json/keyframes_output"

        resultados = chatbot.analizar_conclusiones_y_guardar(keyframes_root)

        output_txt_path = "YOLO/conclusiones_generadas.txt"

        with open(output_txt_path, "w", encoding="utf-8") as f:
            for imagen, data in resultados.items():
                f.write(f"Conclusión para {imagen}:\n")
                if 'error' in data:
                    f.write(f"{data}\n")
                else:
                    f.write(f"{json.dumps(data, indent=2)}\n")
                f.write("-" * 50 + "\n")