import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.yolo_dataset import YOLODataset
from models.base_model import load_model

# -----------------------------
# 설정
# -----------------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_classes = 2  # 배경 + 소시지 클래스
model_type = "fasterrcnn"
model_name = "resnet50_fpn"
batch_size = 2
num_epochs = 5
learning_rate = 1e-4

# -----------------------------
# 데이터셋
# -----------------------------
train_dataset = YOLODataset(
    img_dir='data/sausage/images',
    label_dir='data/sausage/labels',
    transforms=transforms.ToTensor()
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))
)

# -----------------------------
# 모델
# -----------------------------
model = load_model(model_type, num_classes=num_classes, model_name=model_name).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# -----------------------------
# 학습 루프
# -----------------------------
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for imgs, targets in train_loader:
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}")

# -----------------------------
# 모델 저장
# -----------------------------
os.makedirs("output", exist_ok=True)
save_path = os.path.join("output", f"{model_name}_sausage.pth")
torch.save(model.state_dict(), save_path)
print(f"모델이 {save_path}에 저장되었습니다.")
