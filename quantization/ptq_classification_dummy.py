import torch
import torchvision
from torchvision import transforms, datasets
import os

# ---------------------------------------------------------
# 1️⃣ 모델 로드 (예: 사전학습된 ResNet18)
# ---------------------------------------------------------
model_fp32 = torchvision.models.resnet18(pretrained=True)
model_fp32.eval()

# ---------------------------------------------------------
# 2️⃣ 데이터셋 (calibration용 - 작게 잡아도 됨)
# ---------------------------------------------------------
calibration_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# 예시로 ImageNet 대신 CIFAR10 사용 (테스트 목적)
calib_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=calibration_transform
)
calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=32, shuffle=False)

# ---------------------------------------------------------
# 3️⃣ Quantization 설정
# ---------------------------------------------------------
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print("✅ QConfig:", model_fp32.qconfig)

# ---------------------------------------------------------
# 4️⃣ Quantization 준비 단계
# ---------------------------------------------------------
torch.quantization.prepare(model_fp32, inplace=True)

# ---------------------------------------------------------
# 5️⃣ Calibration: 실제 데이터로 통과시켜 통계 수집
# ---------------------------------------------------------
with torch.inference_mode():
    for i, (images, _) in enumerate(calib_loader):
        if i >= 10:  # 샘플 10배치 정도면 충분
            break
        model_fp32(images)

# ---------------------------------------------------------
# 6️⃣ Quantization 변환
# ---------------------------------------------------------
quantized_model = torch.quantization.convert(model_fp32.eval(), inplace=False)
print("✅ Quantization complete!")

# ---------------------------------------------------------
# 7️⃣ 저장
# ---------------------------------------------------------
os.makedirs("./quantized_models", exist_ok=True)
torch.save(quantized_model.state_dict(), "./quantized_models/resnet18_int8.pth")

# ---------------------------------------------------------
# 8️⃣ 크기 비교
# ---------------------------------------------------------
torch.save(model_fp32.state_dict(), "./quantized_models/resnet18_fp32.pth")

fp32_size = os.path.getsize("./quantized_models/resnet18_fp32.pth") / 1024 / 1024
int8_size = os.path.getsize("./quantized_models/resnet18_int8.pth") / 1024 / 1024
print(f"FP32: {fp32_size:.2f} MB → INT8: {int8_size:.2f} MB")

# ---------------------------------------------------------
# 9️⃣ 테스트 (정상 추론 확인)
# ---------------------------------------------------------
x = torch.randn(1, 3, 224, 224)
out_fp32 = model_fp32(x)
out_int8 = quantized_model(x)
print(f"✅ Output diff: {(out_fp32 - out_int8).abs().mean():.4f}")
