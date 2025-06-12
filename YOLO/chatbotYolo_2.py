from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
import json

class Chatbot_YOLO_JSON():
    def __init__(self):
        self.api_key = self.load_api_key()
        os.environ['OPENAI_API_KEY'] = self.api_key
        self.llm = self.initialize_llm()

    def load_api_key(self):
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv('OPENAI_API_KEY')

    def initialize_llm(self):
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Función para hacer un análisis estructurado del JSON
    def summarize_detections(self, json_path: str) -> dict:
        # Leer JSON
        with open(json_path, "r") as file:
            all_detections = json.load(file)

        # Diccionario para guardar los resúmenes
        summaries = {}

        # Recorrer cada imagen y generar el resumen
        for image_name, detections in all_detections.items():
            if not detections:
                summaries[image_name] = {
                    "total_objects": 0,
                    "counts": {},
                    "classes_detected": []
                }
                continue

            # Prompt estructurado
            prompt = (
                f"A partir de las siguientes detecciones en la imagen '{image_name}', "
                "genera un resumen estructurado en formato JSON. Agrupa los objetos por tipo, "
                "cuenta cuántos hay de cada uno, y proporciona una lista única de las clases detectadas. "
                "Responde solo con JSON válido sin bloques ```json``` ni comentarios."
                "No incluyas ninguna explicación, solo el JSON.\n\n"
                f"{json.dumps(detections, indent=2)}"
            )

            msg = HumanMessage(content=prompt)
            response = self.llm.invoke([msg])

            try:
                summaries[image_name] = json.loads(response.content)
            except json.JSONDecodeError:
                summaries[image_name] = {
                    "error": "No se pudo interpretar la respuesta como JSON.",
                    "raw_response": response.content
                }

        return summaries

    #Función para describir las detecciones en el JSON
    def describe_detections(self, json_path: str) -> dict:
        with open(json_path, "r") as file:
            all_detections = json.load(file)

        summaries = {}

        for image_name, detections in all_detections.items():
            if not detections:
                summaries[image_name] = {"objects": {}, "unique_classes": []}
                continue

            # Generar prompt estructurado
            prompt = (
                f"Estas son las detecciones de la imagen '{image_name}'. "
                "Quiero que analices cuántos objetos hay de cada clase, "
                "y generes un resumen en formato JSON estructurado, SIN usar bloques de código (nada de ```json).\n\n"
                "Ejemplo del formato esperado:\n"
                "{\n"
                "  \"objects\": {\n"
                "    \"person\": {\n"
                "      \"count\": 2,\n"
                "      \"instances\": [\n"
                "        {\"confidence\": 0.88, \"box\": {\"x1\":..., \"y1\":..., \"x2\":..., \"y2\":...}}\n"
                "      ]\n"
                "    }, ...\n"
                "  },\n"
                "  \"unique_classes\": [\"person\", \"bicycle\"]\n"
                "}\n\n"
                "Datos detectados:\n"
            )

            for idx, obj in enumerate(detections, 1):
                clase = obj['class']
                conf = obj['confidence']
                box = obj['box']
                prompt += f"{idx}. Objeto: {clase}, confianza: {conf:.2f}, box: ({box['x1']}, {box['y1']}) a ({box['x2']}, {box['y2']})\n"

            msg = HumanMessage(content=prompt)
            response = self.llm.invoke([msg])
            response_text = response.content.strip()

            # Limpieza del bloque de código si viene con ```
            if response_text.startswith("```"):
                response_text = response_text.strip("`")
                lines = response_text.splitlines()
                if lines[0].startswith("json"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "":
                    lines = lines[:-1]
                response_text = "\n".join(lines)

            try:
                summaries[image_name] = json.loads(response_text)
            except json.JSONDecodeError:
                summaries[image_name] = {
                    "error": "No se pudo interpretar la respuesta como JSON.",
                    "raw_response": response.content
                }

        return summaries

    def analizar_conclusiones_por_imagen(self, json_path: str) -> dict:
        with open(json_path, "r") as file:
            all_data = json.load(file)

        resultados = {}

        for image_name, data in all_data.items():
            detections = data.get("detections", [])
            collisions = data.get("collisions", [])

            prompt = (
                f"Analiza los datos de la imagen '{image_name}'. "
                "Primero, resume cuántos objetos hay por clase. Luego, analiza si hay colisiones y qué clases están involucradas. "
                "Finalmente, redacta una conclusión breve sobre lo que está pasando en la imagen. "
                "Entrega el resultado en el siguiente formato JSON (sin bloques de código):\n\n"
                "{\n"
                "  \"resumen\": {\"person\": 3, \"car\": 1},\n"
                "  \"colisiones\": [\n"
                "     {\"entre\": [0, 1], \"iou\": 0.3, \"clases_involucradas\": [\"person\", \"person\"]}\n"
                "  ],\n"
                "  \"conclusion\": \"Parece que hay un grupo de personas cerca unas de otras.\"\n"
                "}\n\n"
                "Este es el contenido:\n"
                f"Detecciones: {json.dumps(detections, indent=2)}\n"
                f"Colisiones: {json.dumps(collisions, indent=2)}"
            )

            msg = HumanMessage(content=prompt)
            response = self.llm.invoke([msg])

            try:
                resultados[image_name] = json.loads(response.content)
            except json.JSONDecodeError:
                resultados[image_name] = {
                    "error": "No se pudo interpretar la respuesta como JSON.",
                    "raw_response": response.content
                }

        return resultados

#CLASE MAIN
if __name__ == "__main__":
    chatbot = Chatbot_YOLO_JSON()
    json_path = "C:/Users/panmo/PycharmProjects/PythonProject/YOLO/runs/resultados_json/todas_las_detecciones.json"
    #resumen = chatbot.summarize_detections(json_path)

    resumen = chatbot.analizar_conclusiones_por_imagen(json_path)

    for imagen, data in resumen.items():
        print(f"\nConclusión para {imagen}:\n{json.dumps(data, indent=2)}\n{'-' * 50}")
