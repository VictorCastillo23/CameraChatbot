from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
import base64
from dotenv import load_dotenv

class Chatbot_YOLO_img():
    def __init__(self):
        self.api_key = self.load_api_key()
        os.environ["OPENAI_API_KEY"] = self.api_key
        self.llm = self.initialize_llm()

    def load_api_key(self) -> str:
        load_dotenv()
        return os.getenv("OPENAI_API_KEY")

    def initialize_llm(self):
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def describe_image(self, image_path: str) -> str:
        # Codificamos la imagen local a base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Creamos el mensaje para gpt-4-vision
        msg = HumanMessage(
            content=[
                {"type": "text", "text": "Describe lo que hay en esta imagen, especialmente las personas detectadas. Enlista todas las características que veas."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )

        # Enviamos el mensaje al modelo
        response = self.llm.invoke([msg])
        return response.content

# Ejemplo de uso
if __name__ == "__main__":
    chatbot = Chatbot_YOLO_img()
    image_path = "C:/Users/panmo/PycharmProjects/PythonProject/YOLO/runs/resultados_yolo/image_6.jpg"
    result = chatbot.describe_image(image_path)
    print(result)