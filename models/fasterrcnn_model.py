import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes, model_name="resnet50_fpn", pretrained=True, **kwargs):
    if model_name == "resnet50_fpn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    elif model_name == "resnet50_fpn_v2":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    elif model_name == "mobilenet_v3_large_fpn":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    elif model_name == "mobilenet_v3_large_320_fpn":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
    else:
        raise ValueError(f"Unknown FasterRCNN model: {model_name}")

    # 헤드 교체
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
