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
"""
#Cargar librerías
from ultralytics import YOLO

#Función main
def main():
    #Descargar modelo base de YOLO
    model = YOLO('yolo11n.pt').to('cuda')
    #Fine-tunning del modelo
    model.train(data="prueba_1.v1i.yolov11/data.yaml", epochs=5)
    #Evaluación del modelo
    model.val()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # ← buena práctica en Windows
    main()