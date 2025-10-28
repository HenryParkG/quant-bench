"""모든 모델의 공통 베이스 클래스"""

"""
모델 공통 베이스 클래스
"""

class BaseModel:
    def __init__(self, name="BaseModel"):
        self.name = name
        print(f"[MODEL] {self.name} 초기화")

    def forward(self, x):
        # 순전파 뼈대
        return x

    def train(self, dataset):
        print(f"[MODEL] {self.name} 학습 중...")
        for i in range(len(dataset)):
            x, y = dataset[i]  # __getitem__ 사용
            out = self.forward(x)
            # dummy loss 계산
        print(f"[MODEL] {self.name} 학습 완료")


    def evaluate(self, dataset):
        print(f"[MODEL] {self.name} 평가 중...")
        correct = 0
        for i in range(len(dataset)):
            x, y = dataset[i]
            pred = self.forward(x) % 2
            if pred == y:
                correct += 1
        acc = correct / len(dataset)
        print(f"[MODEL] {self.name} 평가 완료, Accuracy={acc}")
        return acc
