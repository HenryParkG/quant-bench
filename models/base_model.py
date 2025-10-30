import importlib

MODEL_MAP = {
    "fasterrcnn": "models.fasterrcnn_model",
    "maskrcnn": "models.maskrcnn_model",
    "keypointrcnn": "models.keypointrcnn_model",
    "retinanet": "models.retinanet_model",
    "ssd": "models.ssd_model",
}

def load_model(model_type: str, num_classes: int, **kwargs):
    if model_type not in MODEL_MAP:
        raise ValueError(f"Unknown model type: {model_type}")

    module = importlib.import_module(MODEL_MAP[model_type])
    return module.get_model(num_classes=num_classes, **kwargs)
