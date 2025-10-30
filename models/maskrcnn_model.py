import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model(num_classes: int, model_name="maskrcnn_resnet50_fpn"):
    if model_name == "maskrcnn_resnet50_fpn":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    elif model_name == "maskrcnn_resnet50_fpn_v2":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
    else:
        raise ValueError(f"Unknown MaskRCNN model: {model_name}")

    # Box head 교체
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = MaskRCNNPredictor(in_features, num_classes, model.roi_heads.mask_predictor.conv5_mask.out_channels)
    return model
