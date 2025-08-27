import time
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import psutil

# Importación de las tres clases desde sus archivos
from chatbot_Json import Chatbot_YOLO_JSON
from chatbot_NoYolo import Chatbot_WITHOUT_YOLO
from chatbot_WithYolo import Chatbot_WITH_YOLO

load_dotenv()

class ComparadorChatbots:
    def __init__(self):
        self.tiempos = {}
        self.memorias = {}

        self.mysql_config = {
            'user': os.getenv('MYSQL_USER'),
            'password': os.getenv('MYSQL_PASSWORD'),
            'host': os.getenv('MYSQL_HOST'),
            'database': os.getenv('MYSQL_DATABASE'),
            'port': int(os.getenv('MYSQL_PORT', 3306)),
        }

    def ejecutar_y_medir_tiempo(self):
        process = psutil.Process(os.getpid())

        print("🧪 Ejecutando Chatbot_YOLO_JSON...")
        t1 = time.time()
        bot = Chatbot_YOLO_JSON(self.mysql_config)
        bot.analizar_conclusiones_y_guardar(
            r"C:/Users/panmo/PycharmProjects/PythonProject/YOLO/runs/resultados_json/todas_las_detecciones_global.json"
        )

        t2 = time.time()
        self.tiempos["YOLO_JSON"] = t2 - t1
        self.memorias["YOLO_JSON"] = bot.promedio_memoria_mb
        print(f"✅ Tiempo YOLO_JSON: {self.tiempos['YOLO_JSON']:.2f} s")
        print(f"🧠 Memoria YOLO_JSON: {self.memorias['YOLO_JSON']:.2f} MB\n")

        print("🧪 Ejecutando Chatbot_WITHOUT_YOLO...")
        mem_before = process.memory_info().rss
        t3 = time.time()
        bot = Chatbot_WITHOUT_YOLO(self.mysql_config)
        bot.analizar_conclusiones_y_guardar(
            r"C:/Users/panmo/PycharmProjects/PythonProject/videos/keyframes_output"
        )
        t4 = time.time()
        self.tiempos["WITHOUT_YOLO"] = t4 - t3
        self.memorias["WITHOUT_YOLO"] = bot.promedio_memoria_mb
        print(f"✅ Tiempo WITHOUT_YOLO: {self.tiempos['WITHOUT_YOLO']:.2f} s")
        print(f"🧠 Memoria WITHOUT_YOLO: {self.memorias['WITHOUT_YOLO']:.2f} MB\n")

        print("🧪 Ejecutando Chatbot_WITH_YOLO...")
        mem_before = process.memory_info().rss
        t5 = time.time()
        bot = Chatbot_WITH_YOLO(self.mysql_config)
        bot.analizar_conclusiones_y_guardar(
            r"C:/Users/panmo/PycharmProjects/PythonProject/YOLO/runs/resultados_json/keyframes_output"
        )
        t6 = time.time()
        self.tiempos["WITH_YOLO"] = t6 - t5
        self.memorias["WITH_YOLO"] = bot.promedio_memoria_mb
        print(f"✅ Tiempo WITH_YOLO: {self.tiempos['WITH_YOLO']:.2f} s")
        print(f"🧠 Memoria WITH_YOLO: {self.memorias['WITH_YOLO']:.2f} MB\n")

    def graficar_tiempos(self):
        nombres = list(self.tiempos.keys())
        valores = list(self.tiempos.values())

        plt.figure(figsize=(10, 6))
        plt.bar(nombres, valores, color=["lightskyblue", "moccasin", "lightgreen"]
                )
        plt.title("Comparison of execution times between Chatbots", fontsize=18)
        plt.xlabel("Chatbot", fontsize=14)
        plt.ylabel("Time (seconds)", fontsize=14)
        for i, v in enumerate(valores):
            plt.text(i, v + 0.5, f"{v:.2f}s", ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig("tiempos_global.png")
        plt.show()

    def graficar_memoria(self):
        nombres = list(self.memorias.keys())
        valores = list(self.memorias.values())

        plt.figure(figsize=(10, 6))
        plt.bar(nombres, valores, color=["lightskyblue", "moccasin", "lightgreen"])
        plt.title("Memory Usage Comparison Between Chatbots", fontsize=18)
        plt.xlabel("Chatbot", fontsize=14)
        plt.ylabel("Memory used (MB)", fontsize=14)
        for i, v in enumerate(valores):
            plt.text(i, v + 0.5, f"{v:.2f} MB", ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig("memoria_global.png")
        plt.show()

#CLASE MAIN
if __name__ == "__main__":
    comparador = ComparadorChatbots()
    comparador.ejecutar_y_medir_tiempo()
    comparador.graficar_tiempos()
    comparador.graficar_memoria()