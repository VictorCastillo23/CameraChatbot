from langchain_openai import ChatOpenAI
import os

class LogEvaluator:
    def __init__(self):
        self.api_key = self.load_api_key()
        self.llm = self.initialize_llm()

    def load_api_key(self) -> str:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv("OPENAI_API_KEY")

    def initialize_llm(self) -> ChatOpenAI:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=self.api_key)

    def evaluate_log(self):
        """Lee el log y genera una evaluación de la calidad de las respuestas"""
        try:
            with open("chat_log.json", "r", encoding="utf-8") as log_file:
                log_data = log_file.read()

            prompt = f"""
            Analiza la siguiente conversación entre un usuario y un chatbot de seguridad_
            {log_data}
            
            Evalúa la claridad, relevancia y precisión de las respuestas del chatbot.
            Sugiere mejoras para hacer las respuestas más útiles y efectivas.
            """

            response = self.llm.invoke(prompt)
            return response.content
        except FileNotFoundError:
            return "No se encontró el archivo log.txt"

if __name__ == "__main__":
    evaluator = LogEvaluator()
    feedback = evaluator.evaluate_log()
    print("Evaluación del chatbot: ")
    print(feedback)