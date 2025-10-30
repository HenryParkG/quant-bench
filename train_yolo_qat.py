import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.ao.quantization import (
    QConfig, default_observer, default_weight_observer, prepare_qat, convert
)
from models.yolo11_model import get_model  # YOLOv11 모델 로딩
from utils.collate_fn import yolo_collate  # 커스텀 collate 함수
# ==============================================
# YOLO Dataset
# ==============================================
class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transforms = transforms
        self.img_files = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        # 실제 구현에서는 PIL.Image.open(img_path)
        img = torch.randn(3, self.img_size, self.img_size)  # placeholder

        label_path = img_path.replace(self.img_dir, self.label_dir).replace('.jpg', '.txt').replace('.png', '.txt')
        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x, y, w, h = map(float, parts)
                        x1 = (x - w / 2) * self.img_size
                        y1 = (y - h / 2) * self.img_size
                        x2 = (x + w / 2) * self.img_size
                        y2 = (y + h / 2) * self.img_size
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(cls))

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

# ==============================================
# YOLOv11 Loss (더미 손실, 리스트 처리 가능)
# ==============================================
class YOLOv11Loss(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, predictions, targets):
        total_loss = 0.0
        for pred in predictions:
            # pred가 list라면 Tensor로 변환
            if isinstance(pred, list):
                pred = torch.stack(pred, dim=0)
            total_loss += pred.abs().mean() * 0.01
        return {'loss': total_loss}

# ==============================================
# QAT 설정
# ==============================================
def setup_quantization(model, apply_qat=True):
    if not apply_qat:
        return model

    qat_qconfig = QConfig(
        activation=default_observer,
        weight=default_weight_observer
    )
    
    unsupported_modules = (nn.MultiheadAttention, nn.Linear)

    def set_qconfig_selective(module, qconfig):
        if isinstance(module, unsupported_modules):
            module.qconfig = None
        elif isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            module.qconfig = qconfig
        else:
            module.qconfig = None
        for child in module.children():
            set_qconfig_selective(child, qconfig)

    # Backbone만 적용
    set_qconfig_selective(model.backbone, qat_qconfig)

    # Neck 일부 제외, Head FP32 유지
    for name, m in model.neck.named_modules():
        m.qconfig = None
    for m in model.head.modules():
        m.qconfig = None

    model = prepare_qat(model, inplace=True)
    return model

# ==============================================
# 한 에포크 학습
# ==============================================
def train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.to(device)
        targets = [t.to(device) for t in targets]

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = compute_yolo_loss(outputs, targets)  # 직접 계산

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")
    return avg_loss
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_yolo_loss(preds, targets, num_classes=2, device="cuda"):
    """
    preds: (B, num_anchors, 5 + num_classes)
        5 = [x, y, w, h, obj]
    targets: list[Tensor] - each (N, 6)
        [batch_idx, class, x, y, w, h]
    """

    # 손실 항목
    bbox_loss = 0.0
    obj_loss = 0.0
    cls_loss = 0.0

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    for i, pred in enumerate(preds):
        # pred: (num_anchors, 5 + num_classes)
        t = targets[i]
        if t.numel() == 0:
            # 대상 없는 이미지 → 객체 존재 확률만 학습
            obj_loss += bce(pred[..., 4], torch.zeros_like(pred[..., 4]))
            continue

        # [x, y, w, h, obj, class_prob...]
        pred_box = pred[..., 0:4]
        pred_obj = pred[..., 4]
        pred_cls = pred[..., 5:]

        # target 정보
        gt_boxes = t[:, 2:6].to(device)
        gt_classes = t[:, 1].long().to(device)

        # bbox regression loss
        bbox_loss += mse(pred_box, gt_boxes)

        # objectness loss (간단히 GT마다 1로)
        obj_target = torch.ones_like(pred_obj)
        obj_loss += bce(pred_obj, obj_target)

        # class loss
        cls_target = F.one_hot(gt_classes, num_classes=num_classes).float()
        cls_loss += bce(pred_cls, cls_target)

    total_loss = bbox_loss + obj_loss + cls_loss

    loss_dict = {
        "box_loss": bbox_loss.item(),
        "obj_loss": obj_loss.item(),
        "cls_loss": cls_loss.item(),
        "total_loss": total_loss
    }

    return total_loss, loss_dict

# ==============================================
# 메인
# ==============================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2
    batch_size = 4
    num_epochs = 10
    lr = 1e-4
    img_size = 640
    apply_quantization = True
    torch.backends.quantized.engine = 'fbgemm'

    train_dataset = YOLODataset(
        img_dir='data/sausage/images',
        label_dir='data/sausage/labels',
        img_size=img_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=yolo_collate
    )


    model = get_model(num_classes=num_classes, variant='n', pretrained=False)
    model = setup_quantization(model, apply_qat=apply_quantization)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = YOLOv11Loss(num_classes=num_classes)

    best_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        # best 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs("output", exist_ok=True)
            torch.save(model.state_dict(), f"output/yolov11n_best.pth")
            print("💾 Best model saved!")

    # INT8 변환
    if apply_quantization:
        model.cpu()
        model.backbone = convert(model.backbone)
        torch.save(model.state_dict(), f"output/yolov11n_qat_int8.pth")
        print("✓ INT8 model saved")

if __name__ == "__main__":
    main()
