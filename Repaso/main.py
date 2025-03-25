from langchain_openai import ChatOpenAI
#ChatOpenAI es un modelo de IA en la nube que permite hacer operaciones en internet
from langchain_ollama import ChatOllama
#ChatOllama es un modelo de IA libre que se ejecuta sobre la PC

""" Probando modelos """
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.5)
# llm = ChatOllama(model='mistral-nemo', temperature=0.5)
response = llm.invoke('Hola')
response.content


""" Chain básica """
from langchain.prompts import PromptTemplate

template = 'Dame un resumen breve sobre {topic}'
prompt = PromptTemplate(input_variables=['topic'], template=template)

# Crear Chain
chain = prompt | llm

# Ejecutar Chain
response = chain.invoke(
    input={'topic': 'Inteligencia Artificial'}
)
response.content


""" SequentialChain """
# Chain 1: Generar título
title_prompt = PromptTemplate(
    input_variables=['topic'],
    template='Genera un título atractivo sobre {topic}.'
)
title_chain = title_prompt | llm

# Chain 2: Crear resumen
summary_prompt = PromptTemplate(
    input_variables=['title'],
    template='Escribe un resumen breve para el título: {title}.'
)
summary_chain = summary_prompt | llm

# SequentialChain
sequential_chain = title_chain | summary_chain

# Ejecutar Chain
response = sequential_chain.invoke(
    input={'topic': 'Vehiculos autonomos'}
)
response.content


""" RAG - Retrieval Chain """
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Lista de documentos para el Index
path = 'C:/Users/panmo/PycharmProjects/PythonProject/Repaso/documents'
pdfs = os.listdir(path)

# Carga de documentos
docs = [PyPDFLoader(f'{path}/{pdf}').load() for pdf in pdfs]
docs_list = [item for sublist in docs for item in sublist]

# Split - Separar documentos en trozos
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)
docs_splits = text_splitter.split_documents(docs_list)

# Vector Store - Index
vector_store = Chroma.from_documents(documents=docs_splits, embedding=OpenAIEmbeddings())

# Retriever - Recuperador de archivos
retriever = vector_store.as_retriever()

# Invocar retriever
question = 'Cuales son los riesgos del juego Pokemon Go?'
retrieved_docs = retriever.invoke(question)

# Prompt
template = """
Eres un especialista sobre el juego Pokemon Go.
Tienes acceso a información para poder contestar de mejor forma las preguntas del usuario.

Contexto:
{context}

Pregunta: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# RAG Chain
rag_chain = {'context': retriever, 'question': RunnablePassthrough()} | prompt | llm

# Ejecutando Chain
response = rag_chain.invoke(question)
response.content

""" Salidas Estructuradas """
from pydantic import BaseModel

# Definir esquema de salida
class Extraction(BaseModel):
    title: str
    description: str
    sources: str

# Definir prompt
prompt = PromptTemplate(input_variables=['topic'], template='Proporciona información sobre {topic}.')

# Modelo con salida estructurada
llm_structured_output = llm.with_structured_output(Extraction)

# Crear Chain
chain = prompt | llm_structured_output

# Ejecutar Chain
response = chain.invoke(
    input={'topic': 'Energía solar'}
)
response
response.title
response.description
response.sources


""" Historial de mensajes """
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage

# Conversación sin memoria
llm.invoke('Hola, soy Charly')
llm.invoke('Recuerdas mi nombre?')

# Conversación con memoria
# Diccionario para almacenar historial
store = {}

# Función ara obtener historial por id
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Crear Chatbot con memoria
chatbot = RunnableWithMessageHistory(llm, get_session_history)

# Ejecutar Chatbot
config = {'configurable': {'session_id': '1'}}

chatbot.invoke(
    input=[HumanMessage(content='Hola, soy Charly')], # Lista de mensajes
    config=config
)
chatbot.invoke(
    input=[HumanMessage(content='Recuerdas mi nombre?')], # Lista de mensajes
    config=config
)

store


""" Salida estructurada con memoria """
from pydantic import BaseModel, Field

# Definir esquema de salida
class Person(BaseModel):
    name: str = Field(..., title='Nombre de la persona')
    age: int = Field(..., title='Edad de la persona')

# Definir prompt
prompt = ChatPromptTemplate(
    [
        ('system', 'Soy un agente que puede recordar tu nombre y edad'),
        ('human', '{input}')
    ]
)

# Modelo con salida estructurada
# inlude_raw=True para incluir la salida sin procesar
llm_structured_output = llm.with_structured_output(Person, include_raw=True)

# Crear Chain
chain = prompt | llm_structured_output

# Probando Chain sin memoria
chain.invoke('Hola, soy Charly y tengo 25 años')
chain.invoke('')

# Crear Chatbot con memoria
chatbot = RunnableWithMessageHistory(
    chain, 
    get_session_history,
    output_messages_key='raw'
    )

# Ejecutar Chatbot
config = {'configurable': {'session_id': '1'}}

chatbot.invoke(
    input=[HumanMessage(content='')], # Lista de mensajes
    config=config
)

store

""" Agente Sencillo LangGraph """
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from IPython.display import display, Image

# Definir estado del Grafo
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Definir tool
@tool
def add_two_numbers(a, b):
    """ Suma dos numeros """
    return a + b

# Lista de Tools
tools = [add_two_numbers]

# LLM con tools
llm_with_tools = llm.bind_tools(tools)
llm_with_tools.invoke('Cuanto es 2 + 3?')

# Nodos
def call_model(state: MessagesState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

tool_node = ToolNode([add_two_numbers])

# Determinar el siguiente nodo
def next_node(state: MessagesState):
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return 'tools'
    return END

# Construir Grafo
builder = StateGraph(MessagesState)

builder.add_node('model', call_model)
builder.add_node('tools', tool_node)

builder.add_edge(START, 'model')
builder.add_conditional_edges('model', next_node, ['tools', END])
builder.add_edge('tools', 'model')

agent = builder.compile()
display(Image(agent.get_graph().draw_mermaid_png()))

# Ejecutar Grafo
response = agent.invoke(
    input={'messages': 'Cuanto es 2 + 3?'}
)

for message in response['messages']:
    message.pretty_print()


""" react_agent """
from langgraph.prebuilt import create_react_agent

# Crear agante
agent = create_react_agent(llm, tools)

# Ejecutar agente
response = agent.invoke(
    input={'messages': 'Cuanto es 2 + 3?'}
)

for message in response['messages']:
    message.pretty_print()


""" Agent OpenAI """
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Prompt
prompt = ChatPromptTemplate(
    [
        ('system', 'Eres un agente que es capaz de sumar dos números con la herramienta "add_two_numbers"'),
        ('human', '{input}'),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ]
)


# Crear agente Runnable - Pasos intermedios
agent = create_openai_tools_agent(llm, tools, prompt)
# agent = create_tool_calling_agent(llm, tools, prompt)

# Agente ejecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Ejecutar agente
agent_executor.invoke(
    input={'input': 'Cuanto es 2 + 3?'}
)

""" Agente con memoria """

# Prompt
prompt = ChatPromptTemplate(
    [
        ('system', 'Eres un agente que es capaz de sumar dos números con la herramienta "add_two_numbers"'),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{input}'),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ]
)

# Crear agente Runnable
agent = create_openai_tools_agent(llm, tools, prompt)
# agent = create_tool_calling_agent(llm, tools, prompt)

# Agente ejecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Agente con memoria
agent_memory = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history'
)

# Ejecutar agente con memoria
config = {'configurable': {'session_id': '100'}}

response = agent_memory.invoke(
    input={'input': '5 y 17'},
    config=config
)

store