from torchvision.datasets import VOCDetection
from .base_dataset import BaseDataset

class VOCDataset(BaseDataset):
    def __init__(self, root, year='2012', image_set='train', transforms=None):
        super().__init__(transforms)
        self.dataset = VOCDetection(root=root, year=year, image_set=image_set)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        img = self.apply_transforms(img)
        return img, target
