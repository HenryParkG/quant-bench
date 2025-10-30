import torchvision
from torchvision.models.detection.ssd import SSDClassificationHead

def get_model(num_classes: int, model_name="ssdlite_mobilenet_v3_large"):
    if model_name == "ssd300_vgg16":
        model = torchvision.models.detection.ssd300_vgg16(weights="DEFAULT")
    elif model_name == "ssdlite320_mobilenet_v3_large":
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights="DEFAULT")
    else:
        raise ValueError(f"Unknown SSD model: {model_name}")

    # Classification head 교체
    model.head.classification_head.num_classes = num_classes
    return model
