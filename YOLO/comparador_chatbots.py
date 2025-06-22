import time
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

# Importación de las tres clases desde sus archivos
from chatbot_Json import Chatbot_YOLO_JSON
from chatbot_NoYolo import Chatbot_WITHOUT_YOLO
from chatbot_WithYolo import Chatbot_WITH_YOLO

load_dotenv()

class ComparadorChatbots:
    def __init__(self):
        self.tiempos = {}

        self.mysql_config = {
            'user': os.getenv('MYSQL_USER'),
            'password': os.getenv('MYSQL_PASSWORD'),
            'host': os.getenv('MYSQL_HOST'),
            'database': os.getenv('MYSQL_DATABASE'),
            'port': int(os.getenv('MYSQL_PORT', 3306)),
        }

    def ejecutar_y_medir_tiempo(self):
        print("🧪 Ejecutando Chatbot_YOLO_JSON...")
        t1 = time.time()
        Chatbot_YOLO_JSON(self.mysql_config).analizar_conclusiones_y_guardar(
            r"C:/Users/panmo/PycharmProjects/PythonProject/YOLO/runs/resultados_json/todas_las_detecciones_global.json"
        )
        t2 = time.time()
        self.tiempos["YOLO_JSON"] = t2 - t1
        print(f"✅ Tiempo YOLO_JSON: {self.tiempos['YOLO_JSON']:.2f} s\n")

        print("🧪 Ejecutando Chatbot_WITHOUT_YOLO...")
        t3 = time.time()
        Chatbot_WITHOUT_YOLO(self.mysql_config).analizar_conclusiones_y_guardar(
            r"C:/Users/panmo/PycharmProjects/PythonProject/videos/keyframes_output"
        )
        t4 = time.time()
        self.tiempos["WITHOUT_YOLO"] = t4 - t3
        print(f"✅ Tiempo WITHOUT_YOLO: {self.tiempos['WITHOUT_YOLO']:.2f} s\n")

        print("🧪 Ejecutando Chatbot_WITH_YOLO...")
        t5 = time.time()
        Chatbot_WITH_YOLO(self.mysql_config).analizar_conclusiones_y_guardar(
            r"C:/Users/panmo/PycharmProjects/PythonProject/YOLO/runs/resultados_json"
        )
        t6 = time.time()
        self.tiempos["WITH_YOLO"] = t6 - t5
        print(f"✅ Tiempo WITH_YOLO: {self.tiempos['WITH_YOLO']:.2f} s\n")

    def graficar_tiempos(self):
        nombres = list(self.tiempos.keys())
        valores = list(self.tiempos.values())

        plt.figure(figsize=(10, 6))
        plt.bar(nombres, valores, color=["blue", "orange", "green"])
        plt.title("Comparación de tiempos de ejecución entre Chatbots")
        plt.xlabel("Chatbot")
        plt.ylabel("Tiempo (segundos)")
        plt.savefig("tiempos_global.png")
        for i, v in enumerate(valores):
            plt.text(i, v + 0.5, f"{v:.2f}s", ha='center', va='bottom')
        plt.tight_layout()
        plt.show()

#CLASE MAIN
if __name__ == "__main__":
    comparador = ComparadorChatbots()
    comparador.ejecutar_y_medir_tiempo()
    comparador.graficar_tiempos()