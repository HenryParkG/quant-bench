import torchvision
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

def get_model(num_classes: int, model_name="retinanet_resnet50_fpn",  pretrained=True, anchor_sizes=None, **kwargs):
    if model_name == "retinanet_resnet50_fpn":
        model = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT" if pretrained else None)
    elif model_name == "retinanet_resnet50_fpn_v2":
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights="DEFAULT" if pretrained else None)
    else:
        raise ValueError(f"Unknown RetinaNet model: {model_name}")

    # Classification head 교체
    if anchor_sizes:
        model.anchor_generator.sizes = anchor_sizes  # anchor size 변경 가능
    model.head.classification_head.num_classes = num_classes
    return model
