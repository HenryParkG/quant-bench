import os
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np


class DetectionConfig:
    """Detection configuration."""
    def __init__(
        self,
        model_type: str = "yolo",
        model_path: str = "output/yolo11n_single_class.pth",
        test_img_folder: str = "data/sausage/test_images",
        score_thresh: float = 0.5,
        iou_thresh: float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
        device: Optional[str] = None
    ):
        self.model_type = model_type
        self.model_path = Path(model_path)
        self.test_img_folder = Path(test_img_folder)
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.input_size = input_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")


class ModelLoader:
    """Factory for loading different detection models."""
    
    @staticmethod
    def load_model(config: DetectionConfig) -> torch.nn.Module:
        """Load model based on configuration."""
        if config.model_type == "fasterrcnn":
            return ModelLoader._load_fasterrcnn(config)
        elif config.model_type == "custom":
            return ModelLoader._load_custom(config)
        elif config.model_type == "yolo":
            return ModelLoader._load_yolo(config)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
    
    @staticmethod
    def _load_fasterrcnn(config: DetectionConfig) -> torch.nn.Module:
        """Load Faster R-CNN model."""
        model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)
        model.load_state_dict(torch.load(config.model_path, map_location=config.device))
        model.to(config.device)
        model.eval()
        return model
    
    @staticmethod
    def _load_custom(config: DetectionConfig) -> torch.nn.Module:
        """Load custom detection model."""
        from models.lite_detector import LiteDetector
        model = LiteDetector()
        model.load_state_dict(torch.load(config.model_path, map_location=config.device))
        model.to(config.device)
        model.eval()
        return model
    
    @staticmethod
    def _load_yolo(config: DetectionConfig) -> torch.nn.Module:
        """Load YOLO model."""
        from models.yolov11 import get_yolo11n, postprocess
        model = get_yolo11n(nc=1)  # single class
        
        ckpt = torch.load("output/yolo11n_single_class.pth", map_location="cpu")
        print(ckpt.keys())
        
        
        model.load_state_dict(torch.load(config.model_path, map_location=config.device))
        model.to(config.device)
        model.eval()
        

        return model


class Detector:
    """Unified detector interface."""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.model = ModelLoader.load_model(config)
        self.transform = transforms.Compose([
            transforms.Resize(config.input_size),
            transforms.ToTensor()
        ])
    
    def predict(self, img_path: Path) -> List[List[float]]:
        """
        Run inference on a single image.
        
        Args:
            img_path: Path to input image
            
        Returns:
            List of bounding boxes [x1, y1, x2, y2]
        """
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        
        input_tensor = self.transform(img).unsqueeze(0).to(self.config.device)
        
        with torch.no_grad():
            if self.config.model_type == "fasterrcnn":
                boxes = self._predict_fasterrcnn(input_tensor)
            elif self.config.model_type == "custom":
                boxes = self._predict_custom(input_tensor)
            elif self.config.model_type == "yolo":
                boxes = self._predict_yolo(input_tensor)
            else:
                raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        # Scale boxes back to original image size
        boxes = self._scale_boxes(boxes, orig_w, orig_h)
        
        return boxes
    
    def _predict_fasterrcnn(self, input_tensor: torch.Tensor) -> List[List[float]]:
        """Predict with Faster R-CNN."""
        outputs = self.model(input_tensor)
        boxes = [
            box.cpu().tolist() 
            for box, score in zip(outputs[0]['boxes'], outputs[0]['scores']) 
            if score > self.config.score_thresh
        ]
        return boxes
    
    def _predict_custom(self, input_tensor: torch.Tensor) -> List[List[float]]:
        """Predict with custom detector."""
        obj_preds, reg_preds = self.model(input_tensor)
        boxes = []
        
        for obj_pred, reg_pred in zip(obj_preds, reg_preds):
            obj_prob = torch.sigmoid(obj_pred[0, 0])
            mask = obj_prob > self.config.score_thresh
            
            if mask.sum() == 0:
                continue
            
            coords = reg_pred[0, :, mask].T
            for xywh in coords:
                x, y, w, h = xywh
                x1 = (x - w/2).item()
                y1 = (y - h/2).item()
                x2 = (x + w/2).item()
                y2 = (y + h/2).item()
                boxes.append([x1, y1, x2, y2])
        
        return boxes
    
    def _predict_yolo(self, input_tensor: torch.Tensor) -> List[List[float]]:
        """Predict with YOLO."""
        from models.yolov11 import postprocess
        
        predictions = self.model(input_tensor)
        detections = postprocess(
            predictions,
            img_shape=self.config.input_size,
            conf_thres=self.config.score_thresh,
            iou_thres=self.config.iou_thresh,
            max_det=300
        )
        
        det = detections[0]  # [N, 6] (x1, y1, x2, y2, conf, cls)
        if len(det) > 0:
            boxes = det[:, :4].cpu().tolist()
        else:
            boxes = []
        
        return boxes
    
    def _scale_boxes(
        self, 
        boxes: List[List[float]], 
        orig_w: int, 
        orig_h: int
    ) -> List[List[float]]:
        """Scale boxes from model input size to original image size."""
        if not boxes:
            return boxes
        
        input_w, input_h = self.config.input_size
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h
        
        scaled_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            scaled_boxes.append([
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y
            ])
        
        return scaled_boxes


class Visualizer:
    """Visualization utilities."""
    
    @staticmethod
    def draw_boxes(
        img_path: Path, 
        boxes: List[List[float]], 
        color: str = "red", 
        width: int = 3,
        show: bool = True,
        save_path: Optional[Path] = None
    ):
        """
        Draw bounding boxes on image.
        
        Args:
            img_path: Path to input image
            boxes: List of boxes [x1, y1, x2, y2]
            color: Box color
            width: Line width
            show: Whether to display image
            save_path: Path to save annotated image
        """
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        for box in boxes:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        
        if save_path:
            img.save(save_path)
        
        if show:
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Detections: {len(boxes)}")
            plt.tight_layout()
            plt.show()


def run_inference(config: DetectionConfig, output_dir: Optional[Path] = None):
    """
    Run inference on all images in test folder.
    
    Args:
        config: Detection configuration
        output_dir: Directory to save annotated images (optional)
    """
    # Create output directory if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    detector = Detector(config)
    
    # Get all image files
    img_files = sorted([
        f for f in config.test_img_folder.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
    ])
    
    print(f"Found {len(img_files)} images in {config.test_img_folder}")
    print(f"Model: {config.model_type}")
    print(f"Device: {config.device}")
    print(f"Score threshold: {config.score_thresh}")
    print("-" * 50)
    
    # Process each image
    for img_path in img_files:
        print(f"Processing {img_path.name}...", end=" ")
        
        # Predict
        boxes = detector.predict(img_path)
        print(f"{len(boxes)} detections")
        
        # Visualize
        save_path = output_dir / img_path.name if output_dir else None
        Visualizer.draw_boxes(
            img_path, 
            boxes, 
            show=not output_dir,  # Only show if not saving
            save_path=save_path
        )
    
    if output_dir:
        print(f"\nAnnotated images saved to {output_dir}")


if __name__ == "__main__":
    # Configuration
    config = DetectionConfig(
        model_type="yolo",
        model_path="output/yolo11n_single_class.pth",
        test_img_folder="data/sausage/test_images",
        score_thresh=0.5,
        iou_thresh=0.45,
        input_size=(640, 640)
    )
    
    # Run inference
    run_inference(config, output_dir="output/predictions")
    
    # Or run without saving (just show)
    # run_inference(config)