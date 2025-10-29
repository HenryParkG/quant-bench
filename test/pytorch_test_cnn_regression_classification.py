import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import numpy as np

# ======================================
# 1️⃣ 간단한 Custom Dataset (랜덤 객체)
# ======================================
class DummyObjectDataset(Dataset):
    def __init__(self, num_samples=50, image_size=(64, 64)):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 랜덤 이미지
        img = np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)

        # 랜덤 bounding box와 label
        x1, y1 = np.random.randint(0, 32, 2)
        x2, y2 = x1 + np.random.randint(16, 32), y1 + np.random.randint(16, 32)
        bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
        label = torch.tensor(1)  # 클래스 1
        
        img = F.to_tensor(img)
        return img, {"boxes": bbox, "labels": label}


# ======================================
# 2️⃣ Detection 모델 정의
# ======================================
class TinyObjectDetector(nn.Module):
    """
    간단한 Object Detection 모델:
    - Backbone CNN으로 feature 추출
    - 두 개의 Head:
        (1) bbox 회귀 (x1, y1, x2, y2)
        (2) 클래스 예측 (object / background)
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))  # [B, 64, 1, 1]
        )
        self.flatten = nn.Flatten()
        self.bbox_head = nn.Linear(64, 4)      # bbox 회귀
        self.cls_head = nn.Linear(64, num_classes)  # 클래스 분류

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.flatten(feat)
        bbox = self.bbox_head(feat)
        cls_logits = self.cls_head(feat)
        return bbox, cls_logits


# ======================================
# 3️⃣ 학습 준비
# ======================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TinyObjectDetector(num_classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
cls_loss_fn = nn.CrossEntropyLoss()
bbox_loss_fn = nn.SmoothL1Loss()

dataset = DummyObjectDataset()
def collate_fn(batch):
    return tuple(zip(*batch))
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# ======================================
# 4️⃣ 학습 루프
# ======================================
for epoch in range(5):
    model.train()
    total_loss = 0
    for imgs, targets in dataloader:
        imgs = imgs.to(device)
        gt_boxes = torch.stack([t["boxes"] for t in targets]).to(device)
        gt_labels = torch.stack([t["labels"] for t in targets]).to(device)

        pred_boxes, pred_logits = model(imgs)
        loss_cls = cls_loss_fn(pred_logits, gt_labels)
        loss_bbox = bbox_loss_fn(pred_boxes, gt_boxes)
        loss = loss_cls + loss_bbox

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/5], Loss: {total_loss/len(dataloader):.4f}")

# ======================================
# 5️⃣ 추론 예시
# ======================================
model.eval()
with torch.no_grad():
    sample_img, _ = dataset[0]
    pred_box, pred_cls = model(sample_img.unsqueeze(0).to(device))
    pred_class = torch.argmax(pred_cls, dim=1).item()
    print("📦 Predicted Box:", pred_box[0].cpu().numpy())
    print("🏷️ Predicted Class:", pred_class)
