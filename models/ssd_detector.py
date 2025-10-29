import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16, ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteHead

# -----------------------------
# SSD 계열 모델 함수
# -----------------------------

def get_ssd_vgg16(num_classes=1, pretrained=True):
    """SSD300 + VGG16"""
    model = ssd300_vgg16(pretrained=pretrained)
    in_channels = model.head.classification_head.num_classes
    model.head.classification_head = SSDLiteHead(in_channels, num_classes)
    return model

def get_ssdlite_mobilenet_v3_large(num_classes=1, pretrained=True):
    """SSDLite320 + MobileNetV3 Large"""
    model = ssdlite320_mobilenet_v3_large(pretrained=pretrained)
    in_channels = model.head.classification_head.num_classes
    model.head.classification_head = SSDLiteHead(in_channels, num_classes)
    return model

def get_ssdlite_mobilenet_v3_small(num_classes=1, pretrained=True):
    """SSDLite320 + MobileNetV3 Small"""
    # torchvision에는 small 버전이 없어서 MobileNetV3 Small 기반 커스텀 head 적용 가능
    model = ssdlite320_mobilenet_v3_large(pretrained=pretrained)
    # head 교체
    in_channels = model.head.classification_head.num_classes
    model.head.classification_head = SSDLiteHead(in_channels, num_classes)
    return model

def get_ssd_custom_backbone(backbone, num_classes=1):
    """커스텀 backbone SSD (VGG나 MobileNet 등)"""
    model = ssdlite320_mobilenet_v3_large(pretrained=False)
    # backbone 교체
    model.backbone = backbone
    in_channels = model.head.classification_head.num_classes
    model.head.classification_head = SSDLiteHead(in_channels, num_classes)
    return model
