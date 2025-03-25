#Pedirle que haga el prompt
from chatbot import Chatbot
import os

def main():
    bot = Chatbot()

    tasks = [
        "Hola. Saludame y dime que puedes hacer.",
        "Cortar el vídeo 'Video_1.mp4' del segundo 10 al 20 a velocidad 1.",
        #"¿Qué ves en la cámara A?",
        #"¿Qué te pregunte?",
        #"Extrae el segmento del video 'Video_1.mp4' entre los segundos 30 y 50 a velocidad 2.",
        #"Monitorea la cámara C.",
        #"Quiero un fragmento de 'Video_1.mp4' del segundo 5 al 15 a velocidad 3.",
        #"Revisa la cámara D.",
        #"Adiós, despidete."
    ]

    for i, task in enumerate(tasks, start=1):
        try:
            print(f"Task {i}: {task}")
            response = bot.invoke_chat(task)
            output_text = response.get("output", "No se recivió ninguna respuesta")
            print(f"Chatbot: {output_text}\n")
        except Exception as e:
            print(f"Chatbot: Hubo un problema: {e}")

    #print("Chatbot: ¡Adiós!")

if __name__ == "__main__":
    main()
