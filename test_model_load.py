from models.base_model import load_model

# FasterRCNN ResNet50
model1 = load_model("fasterrcnn", num_classes=3, model_name="resnet50_fpn_v2")

# MaskRCNN
model2 = load_model("maskrcnn", num_classes=3, model_name="maskrcnn_resnet50_fpn_v2")

# KeypointRCNN
model3 = load_model("keypointrcnn", num_classes=3)

# RetinaNet
model4 = load_model("retinanet", num_classes=3, model_name="retinanet_resnet50_fpn_v2")

# SSD
model5 = load_model("ssd", num_classes=3, model_name="ssdlite320_mobilenet_v3_large")

print("model loaded successfully")