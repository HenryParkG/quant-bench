import torch
from torch.utils.data import DataLoader
from datasets.single_class_dataset import SingleClassDetectionDataset
from models.faster_rcnn import get_faster_rcnn_model
from models.yolov11 import get_yolo_model_local_n
from utils.collate_fn import collate_fn
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
dataset = SingleClassDetectionDataset("data/sausage/images", "data/sausage/labels", img_size=320)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Model
# model = get_faster_rcnn_model(num_classes=2).to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

model = get_yolo_model_local_n().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for imgs, targets in dataloader:
        imgs = [img.to(device) for img in imgs]
        targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item():.4f}")

# Save
os.makedirs("output", exist_ok=True)
torch.save(model.state_dict(), "output/faster_rcnn_single_class.pth")
