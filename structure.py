import os

# 각 파일별 기본 docstring/내용
docstrings = {
    "__init__.py": "\"\"\"패키지 초기화 파일\"\"\"",
    "loader.py": "\"\"\"데이터셋 로더 및 전처리 모듈\"\"\"",
    "transforms.py": "\"\"\"데이터 증강 및 변환 모듈\"\"\"",
    "coco_loader.py": "\"\"\"COCO 데이터셋 로더\"\"\"",
    "voc_loader.py": "\"\"\"VOC 데이터셋 로더\"\"\"",
    "imagenet_loader.py": "\"\"\"ImageNet 데이터셋 로더\"\"\"",
    "base_model.py": "\"\"\"모든 모델의 공통 베이스 클래스\"\"\"",
    "resnet.py": "\"\"\"ResNet 모델 정의\"\"\"",
    "mobilenet.py": "\"\"\"MobileNet 모델 정의\"\"\"",
    "custom_model.py": "\"\"\"사용자 정의 모델\"\"\"",
    "train.py": "\"\"\"학습 루프\"\"\"",
    "eval.py": "\"\"\"평가 루프\"\"\"",
    "ptq.py": "\"\"\"Post-Training Quantization 모듈\"\"\"",
    "qat.py": "\"\"\"Quantization-Aware Training 모듈\"\"\"",
    "utils.py": "\"\"\"양자화 관련 유틸리티\"\"\"",
    "runner.py": "\"\"\"벤치마크 실행 스크립트\"\"\"",
    "report.py": "\"\"\"결과 리포트 생성\"\"\"",
    "logger.py": "\"\"\"로그 관리 모듈\"\"\"",
    "config.py": "\"\"\"실험 설정 및 하이퍼파라미터\"\"\"",
    "metrics.py": "\"\"\"평가 지표 계산\"\"\"",
    "download_datasets.py": "\"\"\"데이터셋 다운로드 스크립트\"\"\"",
    "preprocess_data.py": "\"\"\"데이터 전처리 스크립트\"\"\"",
    "main.py": "\"\"\"플랫폼 진입점\"\"\"",
    "exp1.yaml": "# Experiment 1 configuration",
    "exp2.yaml": "# Experiment 2 configuration",
    "requirements.txt": "# 필요한 패키지 목록"
}

# 폴더/파일 구조 정의
structure = {
    "benchmark_platform": {
        "datasets": {
            "custom": ["__init__.py", "loader.py", "transforms.py"],
            "public": ["__init__.py", "coco_loader.py", "voc_loader.py", "imagenet_loader.py"]
        },
        "models": ["__init__.py", "base_model.py", "resnet.py", "mobilenet.py", "custom_model.py"],
        "trainers": ["__init__.py", "train.py", "eval.py"],
        "quantization": ["__init__.py", "ptq.py", "qat.py", "utils.py"],
        "benchmarks": ["__init__.py", "runner.py", "report.py"],
        "utils": ["__init__.py", "logger.py", "config.py", "metrics.py"],
        "experiments": ["exp1.yaml", "exp2.yaml"],
        "scripts": ["download_datasets.py", "preprocess_data.py"],
        "requirements.txt": None,
        "main.py": None
    }
}

def create_structure(base_path, struct):
    for name, content in struct.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        elif isinstance(content, list):
            os.makedirs(path, exist_ok=True)
            for f in content:
                file_path = os.path.join(path, f)
                if not os.path.exists(file_path):
                    with open(file_path, 'w', encoding='utf-8') as fp:
                        fp.write(docstrings.get(f, ""))
        elif content is None:
            if not os.path.exists(path):
                with open(path, 'w', encoding='utf-8') as fp:
                    fp.write(docstrings.get(name, ""))

if __name__ == "__main__":
    create_structure(".", structure)
    print("Git 레포용 플랫폼 구조 생성 완료! (기존 파일은 유지됩니다)")
