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

    def describe_detections(self, json_path: str) -> str:
        #Leer JSON
        with open(json_path, "r") as file:
            detections = json.load(file)

        #Convertir JSON a texto
        prompt = "Describe las siguientes detecciones como si tu estuvieras observado una imagen. Explica lo que ves con un lenguaje natural, como si estuviera contandole a alguien más lo que ves.\n\n"
        prompt += "Detecciones:\n"

        for idx, obj in enumerate(detections, 1):
            clase = obj['class']
            conf = obj['confidence']
            box = obj['box']
            prompt += f"{idx}. Objeto: {clase}, confianza: {conf:.2f}, coordenadas: ({box['x1']}, {box['y1']}) a ({box['x2']}, {box['y2']})\n"

        #Enviar prompt al modelo
        msg = HumanMessage(content=prompt)
        response = self.llm.invoke([msg])
        return response.content

if __name__ == "__main__":
    chatbot = Chatbot_YOLO_JSON()
    json_path = "C:/Users/panmo/PycharmProjects/PythonProject/YOLO/runs/resultados_json/detecciones.json"
    response = chatbot.describe_detections(json_path)
    print(response)