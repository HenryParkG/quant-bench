from torchvision.datasets import CocoDetection
from .base_dataset import BaseDataset

class CocoDataset(BaseDataset):
    def __init__(self, root, annFile, transforms=None):
        super().__init__(transforms)
        self.dataset = CocoDetection(root=root, annFile=annFile)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        img = self.apply_transforms(img)
        return img, target
