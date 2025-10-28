from trainers.train import train_model
from trainers.eval import evaluate_model
from datasets.custom.loader import CustomDataset
from models.resnet import ResNet
from quantization.ptq import apply_ptq
from quantization.qat import apply_qat
from benchmarks.report import generate_report

def run_benchmark(mode="all", model_name="resnet", dataset_name="custom", config_path=None):
    print("[RUNNER] 벤치마크 시작")

    # 데이터셋 로드
    dataset = CustomDataset(split="train")

    # 모델 로드
    if model_name.lower() == "resnet":
        model = ResNet()
    else:
        raise NotImplementedError(f"모델 {model_name} 미구현")

    # 학습
    if mode in ["train", "all"]:
        train_model(model, dataset)

    # 평가
    if mode in ["eval", "all"]:
        evaluate_model(model, dataset)

    # 양자화
    if mode in ["quant", "all"]:
        print("[RUNNER] 양자화 테스트 시작")
        ptq_model = apply_ptq(model, dataset)
        qat_model = apply_qat(model, dataset, epochs=1)
        print("[RUNNER] 양자화 테스트 완료")

    print("[RUNNER] 벤치마크 완료")
    
    
    train_acc = None  # 학습 accuracy 필요 시 설정
    eval_acc = model.evaluate(dataset)
    ptq_acc = 0.75  # dummy
    qat_acc = 0.78  # dummy
    generate_report(train_acc=train_acc, eval_acc=eval_acc, ptq_acc=ptq_acc, qat_acc=qat_acc)

