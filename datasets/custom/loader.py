"""데이터셋 로더 및 전처리 모듈"""

class CustomDataset:
    def __init__(self, split="train"):
        self.split = split
        print(f"[DATASET] CustomDataset 초기화: {split} split")
        self.data = [i for i in range(10)]  # 임시 데이터

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = x % 2  # 임시 라벨
        return x, y