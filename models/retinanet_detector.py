# Focal Loss for Dense Object Detection

import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetHead

def get_retinanet(num_classes=2, pretrained=True):
    model = retinanet_resnet50_fpn(pretrained=pretrained)
    in_channels = model.head.classification_head.conv[0].in_channels
    model.head.classification_head = RetinaNetHead(in_channels,num_classes)
    return model
