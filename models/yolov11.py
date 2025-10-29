import torch
from ultralytics import YOLO


def get_yolo_model_local_n(model_path='pretrained_model/yolo11n.pt', device='cpu'):
    model = YOLO(model_path)  # 이미 저장된 weight 로드
    model.fuse()              # 추론용 fusion
    model.to(device)
    return model

def get_yolo_model_local_s(model_path='pretrained_model/yolo11s.pt', device='cpu'):
    model = YOLO(model_path)  # 이미 저장된 weight 로드
    model.fuse()              # 추론용 fusion
    model.to(device)
    return model

def get_yolo_model_local_m(model_path='pretrained_model/yolo11m.pt', device='cpu'):
    model = YOLO(model_path)  # 이미 저장된 weight 로드
    model.fuse()              # 추론용 fusion
    model.to(device)
    return model

def get_yolo_model_local_l(model_path='pretrained_model/yolo11l.pt', device='cpu'):
    model = YOLO(model_path)  # 이미 저장된 weight 로드
    model.fuse()              # 추론용 fusion
    model.to(device)
    return model

def get_yolo_model_local_x(model_path='pretrained_model/yolo11x.pt', device='cpu'):
    model = YOLO(model_path)  # 이미 저장된 weight 로드
    model.fuse()              # 추론용 fusion
    model.to(device)
    return model


## Debugging 용 코드 - 정상 동작
# if __name__ == "__main__":
#     import numpy as np
#     from PIL import Image

#     DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#     # 더미 이미지 생성 (320x320, RGB)
#     dummy_img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
#     dummy_img = Image.fromarray(dummy_img)

#     # 모델 로드 (예: YOLO small)
#     model = get_yolo_model_local_s(device=DEVICE)

#     # 추론
#     results = model(dummy_img)

#     # 결과 확인
#     print(results)
#     results[0].show()           