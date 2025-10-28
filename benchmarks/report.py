"""결과 리포트 생성"""

"""
report.py
학습/평가/양자화 결과를 기록하고 간단히 시각화
"""

import csv
import os
import matplotlib.pyplot as plt

REPORT_DIR = "benchmarks/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

def save_results(results, filename="results.csv"):
    """
    결과를 CSV로 저장
    Args:
        results (list of dict): [{'mode': 'train', 'accuracy': 0.8, 'note': 'PTQ 전'}, ...]
        filename (str): 파일명
    """
    filepath = os.path.join(REPORT_DIR, filename)
    keys = results[0].keys()
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"[REPORT] 결과 CSV 저장: {filepath}")

def plot_accuracy(results, filename="accuracy.png"):
    """
    Accuracy 결과를 간단히 그래프로 시각화
    Args:
        results (list of dict)
    """
    modes = [r['mode'] for r in results if 'accuracy' in r]
    accs = [r['accuracy'] for r in results if 'accuracy' in r]
    
    plt.figure(figsize=(6,4))
    plt.bar(modes, accs, color='skyblue')
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")
    plt.ylim(0,1)
    
    filepath = os.path.join(REPORT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"[REPORT] Accuracy 그래프 저장: {filepath}")

def generate_report(train_acc=None, eval_acc=None, ptq_acc=None, qat_acc=None):
    """
    전체 보고서 생성
    """
    results = []
    if train_acc is not None:
        results.append({"mode": "train", "accuracy": train_acc, "note": "학습 완료"})
    if eval_acc is not None:
        results.append({"mode": "eval", "accuracy": eval_acc, "note": "학습 모델 평가"})
    if ptq_acc is not None:
        results.append({"mode": "ptq", "accuracy": ptq_acc, "note": "PTQ 모델"})
    if qat_acc is not None:
        results.append({"mode": "qat", "accuracy": qat_acc, "note": "QAT 모델"})
    
    if results:
        save_results(results)
        plot_accuracy(results)
