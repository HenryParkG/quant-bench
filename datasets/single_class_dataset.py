import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class SingleClassDetectionDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=320):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        
        # 라벨 로드
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, base+".txt")
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.read().strip().splitlines()
            for l in lines:
                if l == "":
                    continue
                cls, xc, yc, bw, bh = map(float, l.split())
                x1 = (xc - bw/2) * self.img_size
                y1 = (yc - bh/2) * self.img_size
                x2 = (xc + bw/2) * self.img_size
                y2 = (yc + bh/2) * self.img_size
                boxes.append([x1,y1,x2,y2])
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}
        return img_tensor, target
