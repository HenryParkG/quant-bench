import os
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# -----------------------------
# 설정
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPE = "fasterrcnn"  # "fasterrcnn" / "custom" / "yolo" 등
MODEL_PATH = "output/faster_rcnn_single_class.pth"
TEST_IMG_FOLDER = "data/sausage/test_images"
SCORE_THRESH = 0.5

# -----------------------------
# 이미지 전처리
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor()
])

# -----------------------------
# 모델 로드 (범용)
# -----------------------------
if MODEL_TYPE == "fasterrcnn":
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

elif MODEL_TYPE == "custom":
    # 커스텀 detection 모델
    from models.lite_detector import LiteDetector  # 학습한 모델 파일
    model = LiteDetector()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

# 필요 시 YOLO 등 다른 모델도 elif 추가 가능

# -----------------------------
# 시각화 함수
# -----------------------------
def visualize(img_path, boxes):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

# -----------------------------
# 테스트 루프
# -----------------------------
for img_file in sorted(os.listdir(TEST_IMG_FOLDER)):
    img_path = os.path.join(TEST_IMG_FOLDER, img_file)
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if MODEL_TYPE == "fasterrcnn":
            outputs = model(input_tensor)
            boxes = [b.cpu().tolist() for b,s in zip(outputs[0]['boxes'], outputs[0]['scores']) if s> SCORE_THRESH]

        elif MODEL_TYPE == "custom":
            obj_preds, reg_preds = model(input_tensor)
            # 후처리: obj_preds > SCORE_THRESH인 곳만 bbox로 변환
            boxes = []
            for o, r in zip(obj_preds, reg_preds):
                obj_prob = torch.sigmoid(o[0,0])
                mask = obj_prob > SCORE_THRESH
                if mask.sum() == 0:
                    continue
                coords = r[0,:,mask].T
                for xywh in coords:
                    x,y,w,h = xywh
                    x1 = x - w/2
                    y1 = y - h/2
                    x2 = x + w/2
                    y2 = y + h/2
                    boxes.append([x1.item(),y1.item(),x2.item(),y2.item()])

    visualize(img_path, boxes)
