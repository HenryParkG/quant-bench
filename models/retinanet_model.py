import torchvision
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

def get_model(num_classes: int, model_name="retinanet_resnet50_fpn"):
    if model_name == "retinanet_resnet50_fpn":
        model = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")
    elif model_name == "retinanet_resnet50_fpn_v2":
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights="DEFAULT")
    else:
        raise ValueError(f"Unknown RetinaNet model: {model_name}")

    # Classification head 교체
    model.head.classification_head.num_classes = num_classes
    return model
