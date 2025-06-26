from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
import json
import mysql.connector
from datetime import datetime
from dotenv import load_dotenv
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import psutil

load_dotenv()


class Chatbot_YOLO_JSON():
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

    def analizar_conclusiones_y_guardar(self, json_path: str) -> dict:
        start_global = time.time()
        narrativas_por_carpeta = defaultdict(list)

        with open(json_path, "r") as file:
            all_data = json.load(file)

        resultados = {}
        tiempos_por_video = {}
        memoria_por_video = {}

        for carpeta, imagenes in all_data.items():
            memoria_maxima = 0
            process = psutil.Process(os.getpid())
            start_video = time.time()
            total_respuestas = 0
            respuestas_validas = 0

            for image_name, data in imagenes.items():
                detections = data.get("detections", [])
                collisions = data.get("collisions", [])

                prompt = (
                    f"Vas a recibir un archivo en formato JSON con datos extraidos por un algoritmo detector de objectos."
                    "Analiza los datos de la imagen '{image_name}' en la carpeta '{carpeta}'. "
                    "CONTEXTO: CÁMARA DE UN CENTRO COMERCIAL EN EL ÁREA DE TECNOLOGÍAS\n"
                    "Devuélveme SOLO un JSON con las siguientes claves:\n"
                    "- general_description: descripción general breve de las clases detectadas.\n"
                    "- objects_list: lista en formato bullet de cantidad de objetos por clase.\n"
                    "- collision_analysis: análisis de colisiones y su posible significado.\n"
                    "- narrative: interpretación narrativa de la escena, en lenguaje natural.\n\n"
                    "Por favor, responde SOLO con el JSON plano, sin marcarlo con ```json ni ningún otro formato.\n\n"
                    "Ejemplo de JSON:\n"
                    "{\n"
                    "  \"general_description\": \"En la imagen 'frame_00012.jpg', se han detectado dos clases de objetos: personas y botellas.\",\n"
                    "  \"objects_list\": \"- Personas: 1\\n- Botellas: 1\",\n"
                    "  \"collision_analysis\": \"Análisis de Colisiones\\nNo se han registrado colisiones...\",\n"
                    "  \"narrative\": \"Interpretación Narrativa de la Escena\\nImaginemos que...\"\n"
                    "}\n\n"
                    f"Detecciones: {json.dumps(detections, indent=2)}\n"
                    f"Colisiones: {json.dumps(collisions, indent=2)}"
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
                    collision_analysis = parsed.get("collision_analysis", "")
                    narrative = parsed.get("narrative", "")

                    if 'narrative' in parsed:
                        narrativas_por_carpeta[carpeta].append(parsed['narrative'])

                    resultados[f"{carpeta}/{image_name}"] = parsed

                    self.guardar_story_bd(
                        carpeta, image_name,
                        general_desc, objects_list,
                        collision_analysis, narrative
                    )
                except json.JSONDecodeError:
                    resultados[f"{carpeta}/{image_name}"] = {
                        "error": "No se pudo interpretar la respuesta como JSON.",
                        "raw_response": response_text
                    }
                    self.guardar_story_bd(
                        carpeta, image_name,
                        "", "", "", response_text
                    )

                memoria_actual = process.memory_info().rss / (1024 ** 2)
                if memoria_actual > memoria_maxima:
                    memoria_maxima = memoria_actual

                print(f"Imagen {carpeta}/{image_name} procesada.")

            end_video = time.time()
            tiempos_por_video[carpeta] = end_video - start_video
            memoria_por_video[carpeta] = memoria_maxima

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

        print(f"\n✅ (JSON) Tiempo total: {time.time() - start_global:.2f} segundos.")

        videos = list(tiempos_por_video.keys())
        tiempos = list(tiempos_por_video.values())
        plt.figure(figsize=(10, 6))
        bars = plt.bar(videos, tiempos, color='skyblue')
        plt.title("Processing time per video (JSON)")
        plt.xlabel("Video (folder)")
        plt.ylabel("Time (seconds)")
        plt.xticks(rotation=90, ha='right')
        for bar, tiempo in zip(bars, tiempos):
            altura = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, altura + 0.5, f"{tiempo:.2f}s", ha='center', va='bottom',
                     rotation=90)
        plt.tight_layout()
        plt.savefig("tiempos_por_video_WITH_JSON.png")
        plt.show()

        videos_memoria = list(memoria_por_video.keys())
        usos_memoria = list(memoria_por_video.values())
        plt.figure(figsize=(10, 6))
        bars_memoria = plt.bar(videos_memoria, usos_memoria, color='lightcoral')
        plt.title("Memory peak per video (JSON)")
        plt.xlabel("Video (folder)")
        plt.ylabel("Memory (MB)")
        plt.xticks(rotation=90, ha='right')
        for bar, mem in zip(bars_memoria, usos_memoria):
            altura = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, altura - 5, f"{mem:.2f} MB", ha='center', va='top', rotation=90,
                     color='black', fontsize=9, fontweight='bold')
        plt.tight_layout()
        plt.savefig("memoria_por_video_WITH_JSON.png")
        plt.show()

        precision_global = (respuestas_validas / total_respuestas) * 100 if total_respuestas > 0 else 0
        print(
            f"\n🎯 Precisión de JSON parseado correctamente: {precision_global:.2f}% ({respuestas_validas}/{total_respuestas})")

        precision_por_video = {video: precision_global for video in videos_memoria}

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            [tiempos_por_video[v] for v in videos_memoria],
            [memoria_por_video[v] for v in videos_memoria],
            c=[precision_por_video[v] for v in videos_memoria],
            cmap='viridis', s=100, alpha=0.7
        )
        plt.colorbar(scatter, label='Accuracy (%)')
        plt.title("Trade-off between Latency, Memory and Accuracy (WITH JSON)")
        plt.xlabel("Time (s)")
        plt.ylabel("Memory (MB)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("compensacion_latencia_memoria_precision_WITH_JSON.png")
        plt.show()

        self.promedio_memoria_mb = sum(memoria_por_video.values()) / len(memoria_por_video) if memoria_por_video else 0
        return resultados

    def execute_chatbot_json(self):
        mysql_config = {
            'user': os.getenv('MYSQL_USER'),
            'password': os.getenv('MYSQL_PASSWORD'),
            'host': os.getenv('MYSQL_HOST'),
            'database': os.getenv('MYSQL_DATABASE'),
            'port': int(os.getenv('MYSQL_PORT', 3306)),
        }

        chatbot = Chatbot_YOLO_JSON(mysql_config=mysql_config)
        json_path = r"C:/Users/panmo/PycharmProjects/PythonProject/YOLO/runs/resultados_json/todas_las_detecciones_global.json"
        resultados = chatbot.analizar_conclusiones_y_guardar(json_path)

        output_txt_path = "conclusiones_generadas.txt"
        with open(output_txt_path, "w", encoding="utf-8") as f:
            for imagen, data in resultados.items():
                f.write(f"Conclusión para {imagen}:\n")
                if 'error' in data:
                    f.write(f"{data}\n")
                else:
                    f.write(f"{json.dumps(data, indent=2)}\n")
                f.write("-" * 50 + "\n")
