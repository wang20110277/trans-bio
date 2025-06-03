import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils.data_processing import read_fasta, one_hot_encode
from models.cnn_model import build_cnn_model
from utils.data_processing import load_hbv_data
from models.cnn_model import build_hbv_cnn_model

# 设置随机种子以确保可重复性
tf.random.set_seed(42)
np.random.seed(42)

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_data(data_dir):
    """加载并预处理数据"""
    # 读取FASTA文件
    fasta_path = os.path.join(data_dir, "raw", "GCF_000004075.3_Cucumber_9930_V3_genomic.fna")
    sequences = read_fasta(fasta_path)

    # One-hot编码
    encoded_sequences = [one_hot_encode(seq) for seq in sequences]

    # 转换为NumPy数组
    X = np.array(encoded_sequences)

    # 生成示例标签（假设是二分类任务）
    y = np.random.randint(2, size=len(sequences))  # 替换为真实标签

    return X, y

def split_data(X, y, test_size=0.2, val_size=0.1):
    """划分训练集、验证集和测试集"""
    from sklearn.model_selection import train_test_split

    # 先划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # 再从训练集中划分验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(config):
    """训练模型"""
    # 加载 HBV 数据
    data_dir = os.path.join("data")
    X, y = load_hbv_data(data_dir)

    # 划分数据集
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size=0.2, val_size=0.1)

    # 构建 HBV 模型
    model = build_hbv_cnn_model(input_shape=config["input_shape"])
    model.compile(optimizer=config["optimizer"], loss=config["loss"], metrics=[config["metrics"]])

    # 设置回调函数
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join("models", "saved_models", "best_model.h5"),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # 训练模型
    history = model.fit(
        X_train,
        y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # 保存最终模型
    model.save(os.path.join("models", "saved_models", "final_model.h5"))

    # 返回训练历史
    return history

if __name__ == "__main__":
    # 加载配置文件
    config_path = os.path.join("config", "config.yaml")
    config = load_config(config_path)

    # 训练模型
    history = train_model(config)

    # 打印训练完成信息
    print("模型训练完成！")