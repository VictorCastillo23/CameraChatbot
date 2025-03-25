from langchain.agents import Tool, initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pTools import buscar_procedimiento_emergencia, buscar_horarios
from dotenv import load_dotenv
import os

# Definición de herramientas (sin espacios en los nombres)
tools = [
    Tool(
        name="buscar_procedimiento_emergencia",
        func=buscar_procedimiento_emergencia,
        description="Útil para buscar procedimientos de seguridad en caso de emergencias como incendios o terremotos."
    ),
    Tool(
        name="buscar_horarios",
        func=buscar_horarios,
        description="Útil para consultar horarios de acceso a diferentes ubicaciones del edificio."
    )
]

# Función para cargar el prompt
def cargar_prompt():
    with open("../prompts.md", "r") as f:
        template = f.read()
    return PromptTemplate(template=template, input_variables=["history", "input"])

# Configuración del modelo y agente
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)

# Inicializa el agente
agent = initialize_agent(
    tools=tools,
    llm=chat_model,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True  # Opcional, muestra información detallada para depuración
)

# Función para guardar las conversaciones
def guardar_conversacion(file_path, user_input, agent_response):
    with open(file_path, "a") as f:
        f.write(f"Usuario: {user_input}\n")
        f.write(f"Guardian: {agent_response}\n\n")

# Ciclo principal del chatbot
def iniciar_chatbot():
    historial = ""
    archivo_memoria = "conversacion.txt"

    print("Guardian: Hola, soy tu guardia de seguridad virtual. ¿En qué puedo ayudarte hoy?")

    while True:
        user_input = input("Usuario: ")
        if user_input.lower() in ["salir", "bye", "adiós"]:
            print("Guardian: Hasta luego. Mantente seguro.")
            break

        # Usa el método invoke con el formato correcto
        respuesta = agent.invoke({"input": user_input, "history": historial})
        print(f"Guardian: {respuesta}")

        guardar_conversacion(archivo_memoria, user_input, respuesta)
        historial += f"Usuario: {user_input}\nGuardian: {respuesta}\n"

# Inicia el chatbot
if __name__ == "__main__":
    iniciar_chatbot()
