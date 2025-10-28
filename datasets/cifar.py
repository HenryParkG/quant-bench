import torch
from torchvision import datasets, transforms
from .base_dataset import BaseDataset

class CIFAR10Dataset(BaseDataset):
    def _load_dataset(self):
        transform = transforms.Compose([
            transforms.Resize(224),  # 모든 모델 호환
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        dataset = datasets.CIFAR10(root="./data", train=self.train, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.train)
        return loader
