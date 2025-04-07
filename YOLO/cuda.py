import torch
print(torch.cuda.is_available())  # Debe devolver True
print(torch.cuda.get_device_name(0))  # Debería decirte "NVIDIA GeForce RTX 4050"
