# quant-bench

`quant-bench`는 다양한 **딥러닝 모델 양자화(Quantization) 기법**을 테스트하고 비교하기 위한 실험용 레포지토리입니다.  
Post-Training Quantization(PTQ), Quantization-Aware Training(QAT) 등을 지원하며, 모델 정확도, 크기, 속도 등 성능을 종합적으로 벤치마킹할 수 있습니다.

---

## 📂 레포지토리 구조

<pre>
quant-bench/
│
├── datasets/ # 데이터셋 처리 스크립트 및 다운로드
│ ├── init.py
│ ├── imagenet.py
│ ├── cifar.py
│ └── utils.py
│
├── models/ # 원본/사전 학습 모델과 래퍼
│ ├── init.py
│ ├── resnet.py
│ ├── vit.py
│ └── utils.py
│
├── quantization/ # 다양한 양자화 기법 구현
│ ├── init.py
│ ├── ptq/ # Post-training Quantization
│ │ ├── minmax.py
│ │ └── histogram.py
│ ├── qat/ # Quantization-aware Training
│ │ ├── fake_quant.py
│ │ └── trainer.py
│ └── utils.py # 공통 함수 (scale, bit config 등)
│
├── benchmarks/ # 실험 스크립트
│ ├── init.py
│ ├── run_experiment.py
│ ├── evaluate.py
│ └── compare_results.py
│
├── configs/ # 실험 설정 YAML 파일
│ ├── resnet_ptq.yaml
│ ├── vit_qat.yaml
│ └── default.yaml
│
├── results/ # 실험 결과 저장
│ ├── logs/
│ └── plots/
│
├── utils/ # 공용 유틸리티
│ ├── logger.py
│ ├── metrics.py
│ └── visualization.py
│
├── requirements.txt
├── setup.py
└── README.md
</pre>

---

## ⚡ 주요 기능

- 다양한 양자화 기법 지원
  - **PTQ(Post-Training Quantization)**: Min-Max, Histogram 기반
  - **QAT(Quantization-Aware Training)**: Fake Quantization 등
- 다양한 모델 지원
  - ResNet, Vision Transformer 등
- 벤치마크 자동화
  - 모델별 정확도, 모델 크기, 연산량 비교
- 재현 가능한 실험 구조
  - YAML 기반 설정 파일로 파라미터 관리
- 결과 시각화
  - Accuracy/Size/Latency 비교 차트 생성

---

## 🛠 설치

```bash
git clone https://github.com/henryparkg/quant-bench.git
cd quant-bench
python -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate  # Windows
pip install -r requirements.txt

```

🏗 사용법
1️⃣ 데이터셋 준비

datasets/ 폴더에서 제공하는 스크립트를 통해 다운로드 및 전처리

from datasets.cifar import get_cifar10
train_loader, test_loader = get_cifar10(batch_size=64)

2️⃣ 모델 로드

models/ 폴더의 모델 래퍼 사용

from models.resnet import ResNet18
model = ResNet18(pretrained=True)

3️⃣ 양자화 적용
PTQ
from quantization.ptq.minmax import apply_minmax_quant
quantized_model = apply_minmax_quant(model, bit_width=8)

QAT
from quantization.qat.trainer import train_qat
qat_model = train_qat(model, train_loader, epochs=10)

4️⃣ 실험 실행

benchmarks/run_experiment.py를 사용하여 YAML 설정 파일 기반 실험

python benchmarks/run_experiment.py --config configs/resnet_ptq.yaml

5️⃣ 결과 확인

results/logs/ → 실험 로그

results/plots/ → Accuracy/Size/Latency 비교 그래프

📊 예제 결과 시각화

Accuracy vs Bit-width

Model Size vs Accuracy

Latency Comparison

자동으로 그래프 생성 가능, 커스텀 시각화는 utils/visualization.py 사용
