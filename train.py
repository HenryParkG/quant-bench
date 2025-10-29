import os
import torch
from torch.utils.data import DataLoader
from utils.collate_fn import collate_fn
from datasets.single_class_dataset import SingleClassDetectionDataset

# ---- 모델 로더 ----
from models.faster_rcnn import get_faster_rcnn_model
from models.yolov11 import get_yolo11n
from models.ssd import get_ssd_vgg16
from models.retinanet import get_retinanet


MODEL_REGISTRY = {
    "fasterrcnn": lambda num_classes: get_faster_rcnn_model(num_classes),
    "yolo11n": lambda num_classes: get_yolo11n(num_classes),
    "ssd": lambda num_classes: get_ssd_vgg16(num_classes),
    "retinanet": lambda num_classes: get_retinanet(num_classes),
}


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for imgs, targets in dataloader:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(imgs, targets)

        # Faster R-CNN / SSD 계열은 loss dict 반환
        if isinstance(outputs, dict) and "loss_classifier" in outputs.keys():
            loss_dict = outputs
            loss = sum(loss for loss in loss_dict.values())
        # YOLO 계열은 loss scalar 반환 (또는 dict)
        elif isinstance(outputs, dict):
            loss = sum(outputs.values())
        elif torch.is_tensor(outputs):
            loss = outputs
        else:
            raise ValueError("Unsupported model output type for training.")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(model_name="yolo11n", dataset_root="data/sausage", num_classes=2, img_size=320, batch_size=2, epochs=5, lr=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Dataset ----
    dataset = SingleClassDetectionDataset(
        os.path.join(dataset_root, "images"),
        os.path.join(dataset_root, "labels"),
        img_size=img_size
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # ---- Model ----
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name '{model_name}', available: {list(MODEL_REGISTRY.keys())}")

    model = MODEL_REGISTRY[model_name](num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # ---- Training ----
    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, dataloader, optimizer, device)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

    # ---- Save ----
    os.makedirs("output", exist_ok=True)
    save_path = f"output/{model_name}_single_class.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Universal Object Detection Trainer (PyTorch)")
    parser.add_argument("--model", type=str, default="yolo11n", help="Model name: fasterrcnn | yolo11n | ...")
    parser.add_argument("--dataset", type=str, default="data/sausage", help="Dataset root path")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes (including background)")
    parser.add_argument("--img_size", type=int, default=320, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        dataset_root=args.dataset,
        num_classes=args.num_classes,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr
    )
