import argparse
import os
import yaml
import sys
import logging

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.training import train_model
from src.models.evaluation import evaluate_model
from src.models.prediction import predict
from src.utils.logger import setup_logger

def load_config(config_path):
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")
        
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def execute_mode(mode, config, logger):
    """根据模式执行相应功能"""
    try:
        if mode == "train":
            train_model(config)
            logger.info("模型训练完成！")
            print("模型训练完成！")
        elif mode == "evaluate":
            evaluate_model(config)
            logger.info("模型评估完成！")
            print("模型评估完成！")
        elif mode == "predict":
            predict(config)
            logger.info("预测完成！")
            print("预测完成！")
    except Exception as e:
        logger.error(f"执行{mode}模式时出错: {e}")
        print(f"执行{mode}模式时出错: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="HBV 靶向药物 AI 预测")
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True, 
        choices=["train", "evaluate", "predict"], 
        help="运行模式: train, evaluate, 或 predict"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="配置文件路径"
    )
    
    args = parser.parse_args()
    
    try:
        # 加载配置文件
        config = load_config(args.config)
        
        # 设置日志系统
        logger = setup_logger(config)
        logger.info(f"启动应用，运行模式: {args.mode}")
        
        # 根据模式执行相应功能
        execute_mode(args.mode, config, logger)
        
    except FileNotFoundError as e:
        print(f"文件未找到错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"程序执行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()