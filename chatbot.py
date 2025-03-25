from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools import tools_list
#streamlit - catalogo de camaras (con rango de fechas)
#simular un funcionamiento con camaras
#log - crear chat para evaluar y chat para crear prompts de prueba
#Generar una "historia" a FUTURO

from dotenv import load_dotenv
import os

store = {}

def load_prompts(file_path="prompts.md"):
    try:
        """Lee el archivo prompts.md y extrae los prompts por sección."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        sections = content.split("# ")  #Separar por etiquetas de secciones
        prompts = {}

        for section in sections:
            lines = section.strip().split("\n")  # Dividir en líneas
            if len(lines) > 1:
                key = lines[0].strip().lower()  # Tomar la primera línea como clave (system, human)
                value = "\n".join(lines[1:]).strip()  # El resto es el prompt
                prompts[key] = value  #Guardar en el diccionario correctamente

        #print("Estos son los prompts -> ", prompts)

        return prompts
    except FileNotFoundError:
        print("No se encontró 'prompts.md'. Se usará un prompt por defecto.")
        return {
            "system": "Eres un asistente de seguridad.",
            "human": "{input}"
        }

class Chatbot:
    """INICIALIZAR CHATBOT, MODELO, AGENTE Y GRAFO"""
    def __init__(self):
        self.api_key = self.load_api_key()
        self.llm = self.initialize_llm()
        self.prompts = load_prompts()  #Cargar prompts desde prompts.md
        self.agent, self.agent_executor = self.initialize_agent()
        self.graph_agent = self.initialize_graph()
        self.generate_graph()

    """CARGAR LA API Key"""
    def load_api_key(self) -> str:
        load_dotenv()
        return os.getenv("OPENAI_API_KEY")

    """CONFIGURAR EL MODELO LLM DE OpenAI"""
    def initialize_llm(self) -> ChatOpenAI:
        #La temperatura es para que el chatbot eche verbo
        #0 = Siempre da una respuesta igual (no muy variada)
        #0.5 = Una respuesta con más variedad
        #>= 1 = Más creatividad y aleatoriedad
        return ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=self.api_key)

    """CREAR EL AGENTE LLM CON HERRAMIENTAS"""
    def initialize_agent(self):
        prompt = ChatPromptTemplate(
            [
                ('system', self.prompts.get('system', 'Eres un asistente de seguridad.')),  #Cargar el prompt del sistema
                ('human', self.prompts.get('human', '{input}')),
                MessagesPlaceholder(variable_name='agent_scratchpad')
            ]
        )

        agent = create_openai_tools_agent(self.llm, tools_list, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools_list, verbose=True)
        return agent, agent_executor

    """CONFIGURAR EL GRAFO"""
    def initialize_graph(self):
        # Creamos el grafo
        # call_moodel es un nodo del grafo que representa la invocación del LLM en ese punto del flujo de ejecución.
        # Recibe los mensajes, genera la respuesta y lo devuelve al grafo.
        def call_model(state: MessagesState):
            messages = state['messages']
            history = state.get('history', [])
            alert_mode = state.get('alert_mode', False)#estado de alerta

            full_messages = history + messages
            response = self.agent_executor.invoke(full_messages)

            #Si se detecta un evento sospechoso, activar alerta y redirigir a 'alert_monitoring'
            if "alert_mode" in response and response["alert_mode"]:
                return {'messages': [response], 'alert_mode': True}

            return {
                'messages': [response],
                'history': full_messages[-10:],#Se guardan los últimos 10 mensajes
                'alert_mode': False
            }

        tool_node = ToolNode(tools_list)

        def alert_monitoring(state: MessagesState):
            return {
                "messages": ["Se ha detectado actividad sospechosa. ¿Qué deseas hacer?\n"
                             "Activar alarma\n"
                             "Grabar evidencia\n"
                             "Notificar a seguridad"]
            }

        def fallback(state: MessagesState):
            return {"messages": ["No estoy entrenado para responder eso. Solo puedo procesar videos y monitorear cámaras."]}

        # Ahora vamos a declarar la función del flujo del grafo.
        # Aquí se decide si se pasa a las tools o al final.
        def next_node(state: MessagesState):
            if state.get('alert_mode', False):  #Redirigir a alert_monitoring si hay alerta
                return 'alert_monitoring'

            messages = state["messages"]
            last_message = messages[-1]

            if last_message.tool_calls:
                return 'tools'
            return END

        builder = StateGraph(MessagesState)

        builder.add_node('model', call_model)
        builder.add_node('tools', tool_node)
        builder.add_node('alert_monitoring', alert_monitoring)
        builder.add_node('fallback', fallback)

        # Ahora le damos un "orden" al grafo.
        # Le indicamos que siempre va a comenzar en model.
        # Luego agregamos la condición en caso de que el prompt sea de otra operación o se finalice el proceso.
        # En caso de que sea la tool, se le indica que debe regresar a model
        builder.add_edge(START, 'model')
        builder.add_conditional_edges('model', next_node, ['tools', 'alert_monitoring', 'fallback', END])
        builder.add_edge('tools', 'model')
        builder.add_edge('alert_monitoring', 'model')  #Después de una alerta, volver al modelo

        return builder.compile()

    """ENVÍA UN MENSAJE AL CHATBOT"""
    def invoke_chat(self, prompt: str):
        try:
            #print(f"Enviando al agente: {prompt}")
            response = self.agent_executor.invoke(input={'input': prompt})
            #print(f"Respuesta recibida: {response}")

            return response
        except Exception as e:
            print(f"Error en AgentExecutor: {e}")
            return {"messages": [f"Error al procesar la solicitud: {str(e)}"]}

    """GENERA Y GUARDA EL GRAFO"""
    def generate_graph(self):
        graph_image = self.graph_agent.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(graph_image)
        print("Grafo guardado como graph.png")

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    history: Annotated[list[AnyMessage], add_messages]#historial

'''
#Modelo LLM de GPT
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, openai_api_key=api_key)

#Prueba del modelo con una pregunta
#response = llm.invoke("Hola, ¿qué eres?")
#print(response.content)

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

"""
llm_with_tools = llm.bind_tools(tools_list)
response = llm_with_tools.invoke("Cut the video 'Video_1.mp4' from second 10 to 20 at speed 1.")
print(response.pretty_print())

response = agent.invoke(
    input = {'messages': 'Cut the video "Video_1.mp4" from second 10 to 20 at speed 1.'}
)

for message in response['messages']:
    message.pretty_print()
"""

"""Agent OpenAI"""
prompt = ChatPromptTemplate(
    [
        ('system', 'Eres un asistente especializado en procesamiento de video. Devuelve los siguientes parámetros en formato JSON:\n'
                   '{{\n'
                   '  "start": 10,\n'
                   '  "end": 20,\n'
                   '  "name": "video.mp4",\n'
                   '  "speed": 2\n'
                   '}}'),
        ('human', '{input}'),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ]
)

# Crear agente Runnable - Pasos intermedios
agent = create_openai_tools_agent(llm, tools_list, prompt)

# Agente ejecutor
agent_executor = AgentExecutor(agent=agent, tools=tools_list, verbose=True)

#Creamos el grafo
#call_moodel es un nodo del grafo que representa la invocación del LLM en ese punto del flujo de ejecución.
#Recibe los mensajes, genera la respuesta y lo devuelve al grafo.
def call_model(state: MessagesState):
    messages = state['messages']
    response = agent_executor.invoke(messages)
    return {'messages': [response]}

tool_node = ToolNode(tools_list)

#Ahora vamos a declarar la función del flujo del grafo.
#Aquí se decide si se pasa a las tools o al final.
def next_node(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return 'tools'
    return END

builder = StateGraph(MessagesState)

builder.add_node('model', call_model)
builder.add_node('tools', tool_node)

#Ahora le damos un "orden" al grafo.
#Le indicamos que siempre va a comenzar en model.
#Luego agregamos la condición en caso de que el prompt sea de otra operación o se finalice el proceso.
#En caso de que sea la tool, se le indica que debe regresar a model
builder.add_edge(START, 'model')
builder.add_conditional_edges('model', next_node, ['tools', END])
builder.add_edge('tools', 'model')

agent = builder.compile()

graph_image = agent.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(graph_image)
print("Grafo guardado como graph.png")

# Ejecutar agente
agent_executor.invoke(
    input={'input': 'Hola. Saludame y dime que puedes hacer.'}
)

agent_executor.invoke(
    input={'input': 'Cortar el vídeo "Video_1.mp4" del segundo 10 al 20 a velocidad 1.'}
)

agent_executor.invoke(
    input={'input': 'Extrae el segmento del video "Video_1.mp4" entre los segundos 30 y 50 a velocidad 2.'}
)

agent_executor.invoke(
    input={'input': 'Quiero un fragmento de "Video_1.mp4" del segundo 5 al 15 a velocidad 3.'}
)
'''