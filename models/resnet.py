import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from .base_model import BaseModel

def ResNet18(num_classes=10):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    return BaseModel(model, num_classes=num_classes)
