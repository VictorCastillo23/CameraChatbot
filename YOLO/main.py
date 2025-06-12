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

#Probar otras versiones de YOLO
#Función para aplicar el modelo YOLO_V8 y hacer un JSON con las imágenes
def detectar_y_guardar_json_multiple(carpeta_imagenes_path, modelo_path='yolov8n.pt', carpeta_salida='runs/resultados_json'):
    # Cargar modelo
    model = YOLO(modelo_path)

    # Crear carpeta si no existe
    os.makedirs(carpeta_salida, exist_ok=True)

    # Diccionario para guardar todas las detecciones
    todas_detecciones = {}

    # Recorrer imágenes
    for nombre_imagen in os.listdir(carpeta_imagenes_path):
        if not nombre_imagen.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        ruta_imagen = os.path.join(carpeta_imagenes_path, nombre_imagen)

        # Ejecutar la predicción con dibujo de cajas
        results = model.predict(
            source=ruta_imagen,
            save=True,
            project='runs',
            name='resultados_json',
            exist_ok=True
        )

        # Obtener detecciones y preparar el JSON
        detecciones = []
        boxes_solas = []

        for box in results[0].boxes:
            clase = results[0].names[int(box.cls)]
            conf = float(box.conf)
            x1, y1, x2, y2 = map(float, box.xyxy[0])

            boxes_solas.append((x1, y1, x2, y2))  # Guardar para colisiones

            detecciones.append({
                "class": clase,
                "confidence": conf,
                "box": {
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                }
            })

        #Detectar colisiones entre pares
        colisiones = []
        for i in range(len(boxes_solas)):
            for j in range(i + 1, len(boxes_solas)):
                iou = calcular_iou(boxes_solas[i], boxes_solas[j])
                if iou > 0.1:
                    colisiones.append({
                        "beetwen": [i, j],
                        "iou": round(iou, 3)
                    })

        # Agregar detecciones al diccionario global
        todas_detecciones[nombre_imagen] = {
            "detections": detecciones,
            "collisions": colisiones,
        }
        print(f'Detecciones procesadas para: {nombre_imagen}')

    # Guardar todas las detecciones en un único JSON
    json_path = os.path.join(carpeta_salida, 'todas_las_detecciones.json')
    with open(json_path, 'w') as f:
        json.dump(todas_detecciones, f, indent=4)

    print(f'Todas las detecciones guardadas en: {json_path}')

#Función para caclcular colisiones
def calcular_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # Coordenadas intersección
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0

    # Áreas individuales
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area
    return iou

# CLASE MAIN
if __name__ == '__main__':
    #detectar_y_guardar_json_multiple('pruebas/')
    detectar_y_guardar_json_multiple('C:/Users/panmo/PycharmProjects/PythonProject/libraryV2/keyframes_yolo')