import random

from langchain_core.tools import tool
from proceso import procesar_video
#from langchain.tools import Tool
import json
from datetime import datetime

"""MANERA 1"""
@tool
def video_processing_tool(task: str) -> str:
    """
       Procesa un video en función de una descripción de tarea en formato JSON.
       Requiere los parámetros 'name', 'start', 'end', y 'speed'.

       Ejemplo de tarea:
       {
           "name": "Video_1.mp4",
           "start": 10,
           "end": 20,
           "speed": 1
       }
    """

    try:
        task_data = json.loads(task)
        video_path = task_data.get("name")
        start_time = task_data.get("start")
        end_time = task_data.get("end")
        speed = task_data.get("speed")

        if not all([video_path, start_time, end_time, speed]):
            return "\nFaltan uno o más parámetros obligatorios: 'name', 'start', 'end', 'speed'."

        result = procesar_video(video_path, start_time, end_time, speed)

        return f"\nVídeo procesado con éxito. Archivo de salida: {result}"

    except json.JSONDecodeError:
        return "\nLa tarea debe contener un formato JSON string correcto."

""" Simulación de Cámaras de Seguridad """
@tool
def explore_cameras(camera: str) -> str:
    """
    Simula la exploración de una cámara de seguridad en la empresa.
    Cámaras disponibles: A, B, C, D.

    Puede recibir un JSON o un string con el nombre de la cámara.

    Ejemplo de JSON:
    {
        "camera": "A"
    }
    """
    alerts = ["Movimiento sospechoso", "Paquete sospechoso", "Sombra en la zona de carga"]

    camera_views = {
        "A": ["Una persona caminando por el pasillo.", "Todo está en calma.", random.choice(alerts)],
        "B": ["Un guardia revisando la zona.", random.choice(alerts), "Dos empleados conversando."],
        "C": ["Una puerta entreabierta sin supervisión.", random.choice(alerts), "Sin actividad inusual."],
        "D": ["Luces parpadeando en el almacén.", "Una sombra se mueve en la zona de carga.", "Cámara con interferencias."]
    }

    try:
        task_data = json.loads(camera)
        camera_id = task_data.get("camera", "").strip().upper()
    except json.JSONDecodeError:
        camera_id = camera.strip().upper()

    if camera_id in camera_views:
        result = random.choice(camera_views[camera])

        if result in alerts:
            return {"message": f"ALERTA: {result} en la cámara {camera}!", "alert_mode": True}
        else:
            return {"message": f"Cámara {camera}: {result}", "alert_mode": False}

    return "Cámara no encontrada. Usa A, B, C o D."

"""Simulacion de resumen de video"""
@tool
def sumarize_video(video: str, length: int = 10) -> str:
    """
        Analiza un video y genera un resumen en texto.

        Parámetros:
        - video: Nombre del archivo de video.
        - length: Número de líneas en el resumen.
        """
    return f"Resumen del video '{video}':\n- [Generar aquí un análisis basado en IA]."

"""Simulación de reportes de video"""
@tool
def incident_report(camera: str, description: str) -> str:
    """
        Genera un reporte de incidente en un archivo JSON.
        """
    incident = {
        "timestamp": datetime.now().isoformat(),
        "camera": camera,
        "description": description
    }

    with open("incident_reports.json", "a") as f:
        json.dump(incident, f)
        f.write("\n")

    return f"Reporte generado para la cámara {camera}."

tools_list = [
    video_processing_tool,
    explore_cameras,
    sumarize_video,
    incident_report
]

"""MANERA 2"""
"""
video_tool = Tool(
    name="video_processing_tool",
    description="A tool to process videos. Accepts a task description as a string or JSON.",
    func=video_processing_tool
)

tools_list = [
    video_tool
]
"""