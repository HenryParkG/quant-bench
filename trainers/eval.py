"""평가 루프"""

def evaluate_model(model, dataset):
    print("[EVAL] 평가 시작")
    acc = model.evaluate(dataset)
    print(f"[EVAL] 평가 종료, Accuracy={acc}")
