import torch
import torch.nn as nn
import torch.optim as optim
from datasets.cifar import CIFAR10Dataset
from models.resnet import ResNet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터
train_dataset = CIFAR10Dataset(batch_size=128, train=True)
train_loader = train_dataset.get_loader()

# 모델
model = ResNet18(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 학습
epochs = 2  # 샘플용 짧게
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# 모델 저장
torch.save(model.state_dict(), "./results/resnet18_cifar10.pth")
