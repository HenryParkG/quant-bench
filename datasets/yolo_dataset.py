import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from .base_dataset import BaseDataset, yolo_to_bbox
import torch

class YOLODataset(BaseDataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        super().__init__(transforms)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.img_files)


    def __getitem__(self, idx):
        img, target = self.load_image_and_target(idx)

        boxes = target["boxes"]
        labels = target["labels"].unsqueeze(1)  # (N,1)

        # YOLO 포맷으로 [class, x, y, w, h] 만들기 (정규화 포함)
        img_w, img_h = img.shape[-1], img.shape[-2]
        xyxy = boxes.clone()
        xywh = torch.zeros_like(xyxy)
        xywh[:, 0] = ((xyxy[:, 0] + xyxy[:, 2]) / 2) / img_w  # x_center
        xywh[:, 1] = ((xyxy[:, 1] + xyxy[:, 3]) / 2) / img_h  # y_center
        xywh[:, 2] = (xyxy[:, 2] - xyxy[:, 0]) / img_w         # w
        xywh[:, 3] = (xyxy[:, 3] - xyxy[:, 1]) / img_h         # h

        targets = torch.cat([labels, xywh], dim=1)
        return img, targets