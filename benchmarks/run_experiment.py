import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from datasets.cifar import CIFAR10Dataset
from models.resnet import ResNet18
from quantization.ptq.minmax import apply_minmax_quant

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


if __name__ == "__main__":
    batch_size = 64
    test_dataset = CIFAR10Dataset(batch_size=batch_size, train=False)
    test_loader = test_dataset.get_loader()
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet18(num_classes=10).to(device)

    # FP32 정확도
    acc_fp32 = evaluate(model, test_loader, device)
    print(f"FP32 Accuracy: {acc_fp32:.4f}")

    # PTQ 적용
    q_model = apply_minmax_quant(model, bit_width=8)
    acc_ptq = evaluate(q_model, test_loader, device)
    print(f"PTQ Accuracy (8-bit): {acc_ptq:.4f}")
