import os
from PIL import Image
from .base_dataset import BaseDataset

class CustomDataset(BaseDataset):
    def __init__(self, img_dir, label_dir, transforms=None, label_format='yolo'):
        super().__init__(transforms)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.label_format = label_format
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, self.img_files[idx].replace('.jpg', '.txt').replace('.png', '.txt'))
        img = Image.open(img_path).convert("RGB")
        img = self.apply_transforms(img)

        # TODO: label_format에 따라 parsing
        target = {"boxes": None, "labels": None}
        return img, target
