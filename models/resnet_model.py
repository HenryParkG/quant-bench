import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes: int, model_name="resnet50_fpn"):
    """
    Faster R-CNN 기반 모델 생성
    model_name options:
        - 'resnet50_fpn'
        - 'resnet50_fpn_v2'
        - 'mobilenet_v3_large_fpn'
        - 'mobilenet_v3_large_320_fpn'
    """
    if model_name == "resnet50_fpn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif model_name == "resnet50_fpn_v2":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)
    elif model_name == "mobilenet_v3_large_fpn":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    elif model_name == "mobilenet_v3_large_320_fpn":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # 헤드 교체 (클래스 수에 맞게)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
