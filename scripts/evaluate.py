import os
import yaml
import numpy as np
import tensorflow as tf
from utils.data_processing import read_fasta, one_hot_encode
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_data(data_dir):
    """加载并预处理数据"""
    # 读取FASTA文件
    fasta_path = os.path.join(data_dir, "raw", "cucumber_genome.fasta")
    sequences = read_fasta(fasta_path)

    # One-hot编码
    encoded_sequences = [one_hot_encode(seq) for seq in sequences]

    # 转换为NumPy数组
    X = np.array(encoded_sequences)

    # 生成示例标签（假设是二分类任务）
    y = np.random.randint(2, size=len(sequences))  # 替换为真实标签

    return X, y

def split_data(X, y, test_size=0.2):
    """划分训练集和测试集"""
    from sklearn.model_selection import train_test_split

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_test, y_test

def evaluate_model(config):
    """评估模型"""
    # 加载数据
    data_dir = os.path.join("data")
    X, y = load_data(data_dir)

    # 划分测试集
    X_test, y_test = split_data(X, y, test_size=0.2)

    # 加载模型
    model_path = os.path.join("models", "saved_models", "best_model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在，请先训练模型。")

    model = tf.keras.models.load_model(model_path)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)  # 将概率值转换为类别

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes)
    recall = recall_score(y_test, y_pred_classes)
    f1 = f1_score(y_test, y_pred_classes)
    roc_auc = roc_auc_score(y_test, y_pred)

    # 打印评估结果
    print("模型评估结果：")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数 (F1 Score): {f1:.4f}")
    print(f"AUC分数 (ROC AUC): {roc_auc:.4f}")

    # 返回评估结果
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }

if __name__ == "__main__":
    # 加载配置文件
    config_path = os.path.join("config", "config.yaml")
    config = load_config(config_path)

    # 评估模型
    evaluation_results = evaluate_model(config)

    # 打印评估完成信息
    print("模型评估完成！")