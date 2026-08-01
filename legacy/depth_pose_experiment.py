import torch
import cv2
import numpy as np
from ultralytics import YOLO

yolo_model = YOLO("../yolov10m.pt")

midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", pretrained=False)

state_dict = torch.load("../models/dpt_hybrid_384.pt", map_location=torch.device("cpu"))
midas.load_state_dict(state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

img_path = "mis_frames/1759356830/96.jpg"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = yolo_model(img_rgb)
boxes = results[0].boxes.xyxy.cpu().numpy()
classes = results[0].boxes.cls.cpu().numpy()

input_batch = transform(img_rgb).to(device)

with torch.no_grad():
    prediction = midas(input_batch)

prediction = torch.nn.functional.interpolate(
    prediction.unsqueeze(1),
    size=img.shape[:2],
    mode="bicubic",
    align_corners=False,
).squeeze()

depth_map = prediction.cpu().numpy()
depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_norm = depth_norm.astype(np.uint8)

for (box, cls_id) in zip(boxes, classes):
    if int(cls_id) == 0:
        x1, y1, x2, y2 = map(int, box)
        roi_depth = depth_map[y1:y2, x1:x2]
        mean_depth = np.mean(roi_depth)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Depth: {mean_depth:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("Detecciones YOLO", img)
cv2.imshow("Mapa de Profundidad", depth_norm)
cv2.waitKey(0)
cv2.destroyAllWindows()
