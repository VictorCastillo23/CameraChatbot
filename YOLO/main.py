"""
#Cargar librerías
from ultralytics import YOLO
from PIL import Image

#Descargar modelo base de YOLO
model = YOLO('yolo11n.pt').to('cuda')

#Procesar detección con el modelo base
img = 'prueba_1.v1i.yolov11/test/images/image_9_jpg.rf.19a358672912f75a6f5c34b46c902b3e.jpg'
results = model.predict(img, device=0)
type(results)
result = results[0]
type(result)
#Image.fromarray(result.plot()[:, :, ::-1]).show()

#Fine-tuning del modelo

Cuando se entrena el modelo se hace por épocas
Cada época toma todo el conjunto de entrenamiento, hace las predicciones, valida las predicciones del modelo con lo que se esperaba,
ajusta sus parámetros y luego valida con los datos de validación.

model.train(data="prueba_1.v1i.yolov11/data.yaml", epochs=5)

#Evaluación del modelo
model.val()

#Cargar librerías
from ultralytics import YOLO
from PIL import Image

#Función main
def main():
    #Descargar modelo base de YOLO
    model = YOLO('yolo11n.pt').to('cuda')
    #Fine-tunning del modelo
    model.train(data="People Detection.v2i.yolov11/data.yaml", epochs=30)
    #Evaluación del modelo
    model.val()

def probar_modelo():

    model = YOLO('./bestmodel/bestmodel4.pt').to('cuda')
    img = 'C:/Users/panmo/PycharmProjects/PythonProject/YOLO/prueba_1.v2i.yolov11/test/images/suggested-d6nk6cKyG0eQvK2VVccm_jpg.rf.85340290e54c62e53862026dab8dbb20.jpg'
    results = model.predict(img, device=0)
    result = results[0]
    Image.fromarray(result.plot()[:, :, ::-1]).show()

    model.predict()

    model = YOLO('yolov8n.pt').to('cuda')  # o el modelo que estés usando

    results = model.predict(
        source='pruebas/image_6.jpg',
        save=True,  # 🔥 fuerza guardar la imagen con cajas
        project='runs',
        name='resultados_yolo',  # carpeta donde se guarda
        exist_ok=True  # no da error si ya existe
    )

    # Mostrar resultados
    results[0].show()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # ← buena práctica en Windows
    #main()
    probar_modelo()
"""
from ultralytics import YOLO
import json
import os

def detectar_y_guardar_json(imagen_path, modelo_path='yolov8n.pt'):
    # Cargar modelo
    model = YOLO(modelo_path)

    # Hacer predicción
    results = model.predict(source=imagen_path, save=True, project='runs', name='resultados_json', exist_ok=True)

    # Obtener primer resultado
    result = results[0]

    # Preparar datos en formato JSON
    detecciones = []
    for box in result.boxes:
        clase = result.names[int(box.cls)]
        conf = float(box.conf)
        x1, y1, x2, y2 = map(float, box.xyxy[0])  # Esquinas de la caja

        detecciones.append({
            "class": clase,
            "confidence": round(conf, 3),
            "box": {
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1)
            }
        })

    # Guardar como JSON
    json_path = 'runs/resultados_json/detecciones.json'
    with open(json_path, 'w') as f:
        json.dump(detecciones, f, indent=4)

    print(f'Detecciones guardadas en: {json_path}')

if __name__ == '__main__':
    detectar_y_guardar_json('pruebas/image_6.jpg')