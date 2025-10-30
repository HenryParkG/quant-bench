import torchvision
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

def get_model(num_classes: int, model_name="keypointrcnn_resnet50_fpn"):
    if model_name != "keypointrcnn_resnet50_fpn":
        raise ValueError(f"Unknown KeypointRCNN model: {model_name}")

    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights="DEFAULT")
    # Box head만 수정
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = KeypointRCNNPredictor(in_features, num_classes)
    return model
