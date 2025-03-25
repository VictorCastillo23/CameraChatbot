import streamlit as st
import json
from chatbot import Chatbot
from dotenv import load_dotenv
import os

#Instancia
bot = Chatbot()

#Configurar la página
st.set_page_config(page_title="Chatbot de Seguridad", layout="wide")

#Título
st.title("Chatbot de Seguridad")

#Menú lateral
st.sidebar.header("Opciones")
action = st.sidebar.radio("Selecciona una acción:", ["Monitorear Cámaras", "Procesar Video", "Ver Registro de Eventos"])

if action == 'Monitorear Cámaras':
    st.subheader("Monitoreo en vivo")

    camera = st.selectbox("Selecciona una cámara:", ["A", "B", "C", "D"])

    if st.button("Monitorear"):
        response = bot.invoke_chat(f"¿Qué ves en la cámara {camera}?")
        output_text = response.get("output", "No response received")
        st.write(f"**Cámara {camera}:** {output_text}")
elif action == 'Procesar Video':
    st.subheader("Procesamiento de Video")

    video_name = st.text_input("Nombre del video (ejemplo: video_1.mp4)")
    start_time = st.number_input("Tiempo de inicio (segundos)", min_value=0)
    end_time = st.number_input("Tiempo de fin (segundos)", min_value=start_time)
    speed = st.slider("Velocidad", 0.5, 4.0, 1.0)

    if st.button("Procesar Video"):
        task = json.dumps({
            "name": video_name,
            "start": start_time,
            "end": end_time,
            "speed": speed
        })

        response = bot.invoke_chat(f"Process this video task: {task}")
        output_text = response.get("output", "No response received.")
        st.write(f"**Resultado:** {output_text}")
"""
elif action == "Ver Registro de Eventos":
    st.subheader("Registro de Eventos de Seguridad")

    try:
        with open("security_log.json", "r") as f:
            events = [json.loads(line) for line in f.readlines()]

        if events:
            for event in reversed(events):
                st.write(f"**{event['timestamp']}** - **Cámara {event['camera']}** - {event['description']}")
        else:
            st.write("No hay eventos registrados.")
    except FileNotFoundError:
        st.write("No hay registros disponibles.")
"""