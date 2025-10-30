import torch
from torch.utils.data import Dataset
import numpy as np

class BaseDataset(Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def apply_transforms(self, img):
        if self.transforms:
            return self.transforms(img)
        return img

# YOLO -> bbox 변환 등 공통 함수도 여기에 넣을 수 있음
def yolo_to_bbox(yolo_labels, img_width, img_height):
    boxes = []
    labels = []
    for label in yolo_labels:
        cls, x_c, y_c, w, h = label
        x1 = (x_c - w / 2) * img_width
        y1 = (y_c - h / 2) * img_height
        x2 = (x_c + w / 2) * img_width
        y2 = (y_c + h / 2) * img_height
        boxes.append([x1, y1, x2, y2])
        labels.append(int(cls))
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)
