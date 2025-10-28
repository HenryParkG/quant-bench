"""플랫폼 진입점"""

import argparse
from benchmarks.runner import run_benchmark

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["train", "eval", "quant", "all"])
    parser.add_argument("--model", default="resnet")
    parser.add_argument("--dataset", default="custom")
    parser.add_argument("--config", default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(mode=args.mode, model_name=args.model, dataset_name=args.dataset, config_path=args.config)
