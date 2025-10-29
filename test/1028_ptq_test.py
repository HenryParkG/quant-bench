import torch
from torch.ao.quantization import (
    get_default_qconfig, prepare, convert
)
import torch.nn as nn
from torchvision import models


# 예제: ResNet18 pretrained
model_fp32 = models.resnet18(pretrained=True)
model_fp32.eval()  # PTQ에서는 eval 모드

# PTQ에서는 observer를 통해 activation scale 계산, PyTorch에서 기본 제공하는 fbgemm (CPU용) 또는 qnnpack 사용
model_fp32.qconfig = get_default_qconfig("fbgemm")

model_prepared = prepare(model_fp32)


dummy_input = torch.randn(1, 3, 224, 224)
_ = model_prepared(dummy_input)

'''
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# 예: CIFAR10 데이터셋에서 일부 샘플
calib_dataset = Subset(datasets.CIFAR10(root="./data", train=True, download=True, transform=transform), range(100))
calib_loader = DataLoader(calib_dataset, batch_size=1, shuffle=False)

for images, _ in calib_loader:
    _ = model_prepared(images)

'''

model_int8 = convert(model_prepared)
model_int8.eval()

# state_dict 저장
torch.save(model_int8.state_dict(), "output/resnet18_int8.pth")

# 전체 모델 저장
torch.save(model_int8, "output/resnet18_int8.pt")