'''
Cargar y analizar prompts.md.
Generar nuevos prompts basados en criterios de claridad, efectividad y completitud.
Proponer mejoras y comparar con los prompts actuales.
'''
from langchain_openai import ChatOpenAI
import os

class PromptGenerator:
    def __init__(self):
        self.api_key = self.load_api_key()
        self.llm = self.initialize_llm()

    def load_api_key(self) -> str:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv("OPENAI_API_KEY")

    def initialize_llm(self) -> ChatOpenAI:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=self.api_key)

    def load_prompts(self, file_path="prompts.md") -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return "No se encontró el archivo de prompts.md"

    def generate_improved_prompts(self, prompts_content: str) -> str:
        """Sugiere mejoras para el archivo prompts.md sin cambiar su intención original."""
        prompt = f"""
        Evalúa la siguiente configuración de prompts para un chatbot de seguridad:

        {prompts_content}

        - ¿Cómo se pueden mejorar las instrucciones sin cambiar su propósito?
        - ¿Hay errores, ambigüedades o áreas que pueden ser más claras?
        - Propón cambios específicos en formato Markdown.

        Devuelve solo las sugerencias de mejora, sin reescribir todo el texto.
        """
        response = self.llm.invoke(prompt)
        return response.content

    def save_prompts(self, new_prompts: str, file_path="prompts.md"):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_prompts)
        print("prompts.md actualizado con mejoras.")

    def evaluate_prompts(self, old_prompts: str, new_prompts: str) -> str:
        """Compara prompts y da retroalimentación sobre mejoras"""
        prompt = f"""
        Compara estos dos conjuntos de prompts para un chatobot de seguridad:
        
        **Prompts Actuales:**
        {old_prompts}
        
        **Nuevos Prompts Generados:**
        {new_prompts}
        
        Evalúa cuál conjunto es más claro, preciso y útil para el chatbot.
        Explica los cambios positivos y su impacto.
        """

        response = self.llm.invoke(prompt)
        return response.content

#EJEMPLO DE EJECUCIÓN
if __name__ == "__main__":
    generator = PromptGenerator()
    current_prompts = generator.load_prompts()
    improved_prompts = generator.generate_improved_prompts(current_prompts)

    print("\n**Nuevos Prompts Sugeridos:**\n")
    print(improved_prompts)

    # Guardar cambios si se desea
    generator.save_prompts(improved_prompts)
