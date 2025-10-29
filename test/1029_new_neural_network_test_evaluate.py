import os
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import transforms

# -----------------------------
# 설정
# -----------------------------
MODEL_PATH = "output/lite_detector_scripted.pt"  # 전체 모델 저장 파일
TEST_IMG_FOLDER = "data/sausage/test_images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 모델 로드
# -----------------------------
model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval()

# -----------------------------
# 이미지 전처리
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((320,320)),
    transforms.ToTensor()
])

# -----------------------------
# Detection 후 시각화 함수
# -----------------------------
def visualize(img_path, boxes):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in boxes:
        x1,y1,x2,y2 = box
        draw.rectangle([x1,y1,x2,y2], outline="red", width=2)
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def postprocess(obj_preds, reg_preds, img_size=320, conf_thresh=0.3):
    boxes = []
    scores = []
    strides = [4, 8, 16]  # 각 feature map stride
    for i, (o, r) in enumerate(zip(obj_preds, reg_preds)):
        obj_prob = torch.sigmoid(o[0,0])
        mask = obj_prob > conf_thresh
        if mask.sum() == 0:
            continue

        H, W = obj_prob.shape
        y_idxs, x_idxs = mask.nonzero(as_tuple=True)
        stride = strides[i]
        for y, x in zip(y_idxs, x_idxs):
            score = obj_prob[y,x].item()
            xc, yc, w, h = r[0,:,y,x]
            # feature map -> 원본 이미지 scale
            xc = xc.item() * stride
            yc = yc.item() * stride
            w = w.item() * stride
            h = h.item() * stride

            x1 = xc - w/2
            y1 = yc - h/2
            x2 = xc + w/2
            y2 = yc + h/2
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
    return boxes, scores



# -----------------------------
# 테스트 이미지 순회
# -----------------------------
for img_file in sorted(os.listdir(TEST_IMG_FOLDER)):
    img_path = os.path.join(TEST_IMG_FOLDER, img_file)
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        obj_preds, reg_preds = model(input_tensor)
        boxes, scores = postprocess(obj_preds, reg_preds, conf_thresh=0.1)

    visualize(img_path, boxes)
                                                             