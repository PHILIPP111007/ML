import torch

K = 10

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DEVICE_CPU = torch.device("cpu")
