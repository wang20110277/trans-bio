import os
import yaml
import numpy as np
import tensorflow as tf
from utils.data_processing import read_fasta, one_hot_encode

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_data(data_path):
    """加载并预处理新数据"""
    # 读取FASTA文件
    sequences = read_fasta(data_path)

    # One-hot编码
    encoded_sequences = [one_hot_encode(seq) for seq in sequences]

    # 转换为NumPy数组
    X = np.array(encoded_sequences)

    return X, sequences

def load_model(model_path):
    """加载训练好的模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在，请先训练模型。")
    return tf.keras.models.load_model(model_path)

def predict(config):
    """使用模型进行预测"""
    # 加载新数据
    data_path = os.path.join("data", "raw", "new_cucumber_genome.fasta")  # 新数据路径
    X, sequences = load_data(data_path)

    # 加载模型
    model_path = os.path.join("models", "saved_models", "best_model.h5")
    model = load_model(model_path)

    # 进行预测
    y_pred = model.predict(X)
    y_pred_classes = (y_pred > 0.5).astype(int)  # 将概率值转换为类别

    # 输出预测结果
    print("预测结果：")
    for i, (seq, pred) in enumerate(zip(sequences, y_pred_classes)):
        print(f"序列 {i + 1}:")
        print(f"  序列: {seq}")
        print(f"  预测类别: {pred[0]}")
        print(f"  预测概率: {y_pred[i][0]:.4f}")
        print()

    # 返回预测结果
    return y_pred, y_pred_classes

if __name__ == "__main__":
    # 加载配置文件
    config_path = os.path.join("config", "config.yaml")
    config = load_config(config_path)

    # 进行预测
    predictions = predict(config)

    # 打印预测完成信息
    print("预测完成！")