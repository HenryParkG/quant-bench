import torch
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection.fcos import FCOSClassificationHead, FCOSRegressionHead

def get_model(num_classes: int):
    """
    FCOS 기반 Object Detection 모델 로드
    """
    model = fcos_resnet50_fpn(pretrained=True)
    # FCOS 헤드를 클래스 수에 맞게 조정
    model.head.classification_head.num_classes = num_classes
    return model
