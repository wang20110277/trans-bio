import argparse
from scripts.train import train_model
from scripts.evaluate import evaluate_model
from scripts.predict import predict
import os
import yaml
def load_config(config_path):
    """加载配置文件"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cucumber Genome Analysis")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "evaluate", "predict"], help="Mode to run: train, evaluate, or predict")
    args = parser.parse_args()
    # 加载配置文件
    config_path = os.path.join("config", "config.yaml")
    config = load_config(config_path)
    if args.mode == "train":
        train_model(config)
    elif args.mode == "evaluate":
        evaluate_model(config)
    elif args.mode == "predict":
        predict(config)