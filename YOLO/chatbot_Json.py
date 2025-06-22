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

    def analizar_conclusiones_y_guardar(self, json_path: str) -> dict:
        start_global = time.time()

        narrativas_por_carpeta = defaultdict(list)

        with open(json_path, "r") as file:
            all_data = json.load(file)

        resultados = {}
        tiempos_por_video = {}

        for carpeta, imagenes in all_data.items():
            for image_name, data in imagenes.items():
                start = time.time()

                detections = data.get("detections", [])
                collisions = data.get("collisions", [])

                prompt = (
                    f"Analiza los datos de la imagen '{image_name}' en la carpeta '{carpeta}'. "
                    "CONTEXTO: CÁMARA DE UN CENTRO COMERCIAL\n"
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

                # Intentar parsear JSON
                try:
                    parsed = json.loads(response_text)
                    general_desc = parsed.get("general_description", "")
                    objects_list = parsed.get("objects_list", "")
                    collision_analysis = parsed.get("collision_analysis", "")
                    narrative = parsed.get("narrative", "")

                    if 'narrative' in parsed:
                        narrativas_por_carpeta[carpeta].append(parsed['narrative'])

                    resultados[f"{carpeta}/{image_name}"] = parsed

                    # Guardar en BD cada campo por separado
                    self.guardar_story_bd(
                        carpeta,
                        image_name,
                        general_desc,
                        objects_list,
                        collision_analysis,
                        narrative
                    )
                except json.JSONDecodeError:
                    # Si no se pudo parsear, guardar todo el texto en narrative y dejar el resto vacío
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

                end = time.time()
                print(f"Imagen {carpeta}/{image_name} procesada en {end - start:.2f} segundos.")

                end = time.time()
                duracion = end - start

                if carpeta not in tiempos_por_video:
                    tiempos_por_video[carpeta] = 0

                tiempos_por_video[carpeta] += duracion

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

            # Puedes imprimirla, guardarla en archivo o incluso crear una tabla nueva para almacenar narrativa por carpeta
            print(f"\nNarrativa general para el video (carpeta: {carpeta}):\n{narrativa_general}\n")

            self.guardar_narrativa_general(carpeta, narrativa_general)

        print(f"\n✅ (JSON) Tiempo total: {time.time() - start_global:.2f} segundos.")

        # Graficar tiempos por video
        videos = list(tiempos_por_video.keys())
        tiempos = list(tiempos_por_video.values())
        plt.figure(figsize=(10, 6))
        bars = plt.bar(videos, tiempos, color='skyblue')

        plt.title("Tiempo de procesamiento por video (JSON)")
        plt.xlabel("Video (carpeta)")
        plt.ylabel("Tiempo (segundos)")
        plt.xticks(rotation=90, ha='right')

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
        plt.savefig("tiempos_por_video_WITH_JSON.png")
        plt.show()

        return resultados

    def execute_chatbot_json(self):
        # Configuración de conexión MySQL
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