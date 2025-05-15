import cv2
print(cv2.cuda.getCudaEnabledDeviceCount())  # Debería devolver ≥1 si tienes GPU compatible
