import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.ao.quantization import QConfig, default_observer, default_weight_observer, prepare_qat, convert
from datasets.yolo_dataset import YOLODataset
from models.base_model import load_model

# -----------------------------
# 설정
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2  # 배경 + 소시지
model_type = "fasterrcnn"
model_name = "resnet50_fpn"
batch_size = 2
num_epochs = 5
learning_rate = 1e-4
torch.backends.quantized.engine = 'fbgemm'  # CPU

# -----------------------------
# 데이터셋 준비
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
# 모델 준비
# -----------------------------
model = load_model(model_type, num_classes=num_classes, model_name=model_name).to(device)

# -----------------------------
# 양자화 적용 여부 설정 (True or False)
# -----------------------------
apply_quantization = False

if apply_quantization:
    # QAT 설정 (Backbone 전용)
    qat_qconfig = QConfig(
        activation=default_observer,
        weight=default_weight_observer  # per-tensor 방식
    )

    def set_qconfig_recursive(module, qconfig):
        module.qconfig = qconfig
        for child in module.children():
            set_qconfig_recursive(child, qconfig)

    set_qconfig_recursive(model.backbone, qat_qconfig)

    print("Before prepare_qat:")
    for name, module in model.backbone.named_modules():
        if hasattr(module, 'stride'):
            print(f"{name}: stride={module.stride}")

    model = prepare_qat(model, inplace=True)

    print("After prepare_qat:")
    for name, module in model.backbone.named_modules():
        if hasattr(module, 'stride'):
            print(f"{name}: stride={module.stride}")

# -----------------------------
# Optimizer 설정
# -----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# -----------------------------
# 학습 루프
# -----------------------------
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for imgs, targets in train_loader:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

# -----------------------------
# 양자화 변환 및 저장
# -----------------------------
model.eval()

if apply_quantization:
    model.backbone = convert(model.backbone)
    save_suffix = "qat_backbone_int8"
else:
    save_suffix = "fp32"

os.makedirs("output", exist_ok=True)
save_path = os.path.join("output", f"{model_name}_sausage_{save_suffix}.pth")
torch.save(model.state_dict(), save_path)
print(f"모델이 {save_path}에 저장되었습니다.")
