import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from .base_dataset import BaseDataset, yolo_to_bbox
import torch
import torchvision.transforms as transforms

class YOLODataset(BaseDataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        super().__init__(transforms)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, self.img_files[idx].rsplit('.', 1)[0] + '.txt')

        img = Image.open(img_path).convert("RGB")
        img_width, img_height = img.size

        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            yolo_labels = np.loadtxt(label_path, ndmin=2)
            boxes, labels = yolo_to_bbox(yolo_labels, img_width, img_height)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        img = self.apply_transforms(img)

        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)

        return img, target
