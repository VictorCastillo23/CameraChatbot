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
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

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

        # Recorrer todas las subcarpetas dentro del directorio keyframes_output
        for carpeta_path in glob.glob(os.path.join(keyframes_root, "*")):
            carpeta = os.path.basename(carpeta_path)
            imagenes = glob.glob(os.path.join(carpeta_path, "*.jpg"))
            memoria_maxima = 0
            process = psutil.Process(os.getpid())
            start_video = time.time()  # ⏱️ Inicio por carpeta
            total_respuestas = 0
            respuestas_validas = 0

            for imagen_path in imagenes:
                image_name = os.path.basename(imagen_path)

                # Prompt sin detecciones, solo basado en la imagen y el contexto
                prompt = (
                    f"Vas a recibir una imagen border boxes para detección de objetos, correspondiente a un fotograma clave de un vídeo."
                    "Analiza la imagen '{image_name}' en la carpeta '{carpeta}', proveniente de un video de vigilancia "
                    "de un centro comercial en el área de tecnologías. Describe lo que podría estar ocurriendo en la escena.\n\n"
                    "Devuélveme SOLO un JSON con las siguientes claves:\n"
                    "- general_description: descripción general de lo que se observa en la imagen.\n"
                    "- objects_list: si es posible, una lista de objetos visibles (aunque no haya datos específicos).\n"
                    "- collision_analysis: si hay movimiento o interacción visible, interpreta posibles colisiones o relaciones.\n"
                    "- narrative: una narrativa libre y creativa de la escena, como si contaras una historia.\n\n"
                    "Por favor, responde SOLO con el JSON plano, sin marcarlo con ```json ni ningún otro formato."
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
                        narrativas_por_carpeta[carpeta].append(narrative)

                    resultados[f"{carpeta}/{image_name}"] = parsed

                    self.guardar_story_bd(
                        carpeta,
                        image_name,
                        general_desc,
                        objects_list,
                        collision_analysis,
                        narrative
                    )
                except json.JSONDecodeError:
                    resultados[f"{carpeta}/{image_name}"] = {
                        "error": "No se pudo interpretar la respuesta como JSON.",
                        "raw_response": response_text
                    }
                    self.guardar_story_bd(
                        carpeta,
                        image_name,
                        "",
                        "",
                        "",
                        response_text
                    )

                print(f"Imagen {carpeta}/{image_name} procesada.")

                memoria_actual = process.memory_info().rss / (1024 ** 2)
                memoria_maxima = max(memoria_maxima, memoria_actual)

            end_video = time.time()  # ⏱️ Fin por carpeta
            tiempos_por_video[carpeta] = end_video - start_video
            memoria_por_video[carpeta] = memoria_maxima

        # Generar narrativa global por carpeta/video
        for carpeta, narrativas in narrativas_por_carpeta.items():
            resumen_prompt = (
                    f"Has recibido una serie de narrativas que corresponden a imágenes extraídas de un video de vigilancia "
                    f"en un centro comercial. A partir de estas narrativas, genera una narrativa general coherente del video.\n\n"
                    f"Lista de narrativas por imagen:\n\n"
                    + "\n\n".join(f"- {n}" for n in narrativas)
                    + "\n\nDevuelve SOLO una narrativa general en texto plano, sin formato JSON."
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
        plt.figure(figsize=(10, 6))
        bars = plt.bar(videos, tiempos, color='skyblue')

        plt.title("Processing time per video (WITH YOLO)")
        plt.xlabel("Video (folder)")
        plt.ylabel("Time (seconds)")
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

        plt.figure(figsize=(10, 6))
        bars_memoria = plt.bar(videos_memoria, usos_memoria, color='salmon')

        plt.title("Memory usage per video (WITH YOLO)")
        plt.xlabel("Video (folder)")
        plt.ylabel("Peak memory (MB)")
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
        plt.colorbar(scatter, label='Accuracy (%)')
        plt.title("Trade-off between Latency, Memory and Accuracy (WITH YOLO)")
        plt.xlabel("Time (s)")
        plt.ylabel("Memory (MB)")
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

        output_txt_path = "conclusiones_generadas.txt"

        with open(output_txt_path, "w", encoding="utf-8") as f:
            for imagen, data in resultados.items():
                f.write(f"Conclusión para {imagen}:\n")
                if 'error' in data:
                    f.write(f"{data}\n")
                else:
                    f.write(f"{json.dumps(data, indent=2)}\n")
                f.write("-" * 50 + "\n")