import torch
from torch.quantization import fuse_modules, get_default_qat_qconfig, prepare_qat, convert
from ultralytics import YOLO
import torch.nn as nn
from pathlib import Path

# -----------------------------
# 1️⃣ 모델 로드
# -----------------------------
base_path = "input/detection/yolo/"
base_model ="yolo11n.pt"
model = YOLO(base_path + base_model)  # pretrained YOLOv11

# -----------------------------
# 2️⃣ Conv+BN+Activation fuse
# -----------------------------
for name, module in model.model.named_children():
    if isinstance(module, nn.Sequential):
        try:
            fuse_modules(module, [['0','1','2']], inplace=True)
        except:
            pass

# -----------------------------
# 3️⃣ QAT 준비
# -----------------------------
model.model.train()  # ✅ 반드시 train 모드
model.model.qconfig = get_default_qat_qconfig('fbgemm')
prepare_qat(model.model, inplace=True)

# -----------------------------
# 4️⃣ Dummy forward (observer calibration)
# -----------------------------
dummy_input = torch.randn(1, 3, 640, 640)
_ = model.model(dummy_input)  # forward pass

# -----------------------------
# 5️⃣ INT8 변환
# -----------------------------
qat_model_int8 = convert(model.model.eval(), inplace=False)

# -----------------------------
# 6️⃣ output 폴더 생성
# -----------------------------
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# -----------------------------
# 7️⃣ 모델 저장 (.pth, .pt)
# -----------------------------

pth_path = output_dir / "yolov11_int8_dummy.pth"
pt_path  = output_dir / "yolov11_int8_dummy.pt"

# state_dict 저장 (.pth)
torch.save(qat_model_int8.state_dict(), pth_path)

# 전체 모델 저장 (.pt)
torch.save(qat_model_int8, pt_path)

print(f"✅ Dummy QAT INT8 모델 저장 완료!\n- {pth_path}\n- {pt_path}")
