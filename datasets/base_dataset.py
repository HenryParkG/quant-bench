import torch
from torch.utils.data import Dataset
import numpy as np
import os
import PIL.Image as Image

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
    def load_image_and_target(self, idx):
            """
            이미지와 YOLO 라벨(.txt) 파일을 읽어서 Tensor로 반환.
            """
            img_path = os.path.join(self.img_dir, self.img_files[idx])
            label_path = os.path.join(
                self.label_dir,
                os.path.splitext(self.img_files[idx])[0] + ".txt"
            )

            # --- 이미지 로드 ---
            img = Image.open(img_path).convert("RGB")
            img_width, img_height = img.size

            # --- 라벨 로드 ---
            if os.path.exists(label_path):
                label_data = np.loadtxt(label_path).reshape(-1, 5)  # [cls, x, y, w, h]
                boxes, labels = yolo_to_bbox(label_data, img_width, img_height)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.long)

            # --- transform 적용 ---
            if self.transforms:
                img = self.transforms(img)

            target = {"boxes": boxes, "labels": labels}
            return img, target
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
