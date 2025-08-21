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
def detectar_y_guardar_json_multiple(model, carpeta_imagenes_path, carpeta_salida):
    # Crear carpeta si no existe
    os.makedirs(carpeta_salida, exist_ok=True)

    # Diccionario para guardar todas las detecciones
    todas_detecciones_carpeta = {}

    # Recorrer imágenes
    for nombre_imagen in os.listdir(carpeta_imagenes_path):
        if not nombre_imagen.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        ruta_imagen = os.path.join(carpeta_imagenes_path, nombre_imagen)

        # Ejecutar la predicción con dibujo de cajas
        results = model.predict(
            source=ruta_imagen,
            save=True,
            project=carpeta_salida,
            name='.',
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
        todas_detecciones_carpeta[nombre_imagen] = {
            "detections": detecciones,
            "collisions": colisiones,
        }
        print(f'Detecciones procesadas para: {nombre_imagen} en carpeta {os.path.basename(carpeta_salida)}')

    return todas_detecciones_carpeta

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

    #IoU
    return inter_area / union_area

# CLASE MAIN
if __name__ == '__main__':
    ruta_base = 'C:/Users/panmo/PycharmProjects/PythonProject/videos/keyframes_output'

    # Carga el modelo una sola vez
    modelo_path = 'yolov8n.pt'
    model = YOLO(modelo_path)

    # Diccionario global con todas las detecciones
    todas_las_detecciones_global = {}

    for root, dirs, files in os.walk(ruta_base):
        imagenes = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not imagenes:
            continue  # si no hay imágenes en esta carpeta, saltar

        # Carpeta relativa a ruta_base (para usar como clave en el JSON)
        carpeta_relativa = os.path.relpath(root, ruta_base)

        print(f'\n🟢 Procesando carpeta: {carpeta_relativa}')

        carpeta_salida = os.path.join('runs', 'resultados_json/keyframes_output', carpeta_relativa)
        print(f'😈 Carpeta salida: {carpeta_salida}')

        # Llamamos tu función que procesa todas las imágenes de esta carpeta
        detecciones_carpeta = detectar_y_guardar_json_multiple(
            model=model,
            carpeta_imagenes_path=root,
            carpeta_salida=carpeta_salida
        )

        # Guardamos bajo la clave de la carpeta relativa
        todas_las_detecciones_global[carpeta_relativa] = detecciones_carpeta

    # Guardar un JSON global con todas las detecciones agrupadas por jerarquía
    json_global_path = os.path.join('runs', 'resultados_json', 'todas_las_detecciones_global.json')
    os.makedirs(os.path.dirname(json_global_path), exist_ok=True)
    with open(json_global_path, 'w') as f:
        json.dump(todas_las_detecciones_global, f, indent=4)

    print(f'\n✅ Todas las detecciones globales guardadas en: {json_global_path}')