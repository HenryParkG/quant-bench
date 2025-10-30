import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from .base_dataset import BaseDataset, yolo_to_bbox

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
        label_path = os.path.join(self.label_dir, self.img_files[idx].replace('.jpg', '.txt').replace('.png', '.txt'))

        img = Image.open(img_path).convert("RGB")
        img_width, img_height = img.size

        if os.path.exists(label_path):
            yolo_labels = np.loadtxt(label_path).reshape(-1, 5)
            boxes, labels = yolo_to_bbox(yolo_labels, img_width, img_height)
        else:
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        img = self.apply_transforms(img)
        return img, target
