import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.data.processing import read_fasta, one_hot_encode, load_hbv_data
from models.cnn_model import build_hbv_cnn_model
from sklearn.model_selection import train_test_split

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def prepare_data(data_dir, config):
    """准备训练数据"""
    try:
        # 读取FASTA文件
        fasta_files = [f for f in os.listdir(os.path.join(data_dir, "raw")) 
                       if f.endswith(('.fasta', '.fna'))]
        
        if not fasta_files:
            raise FileNotFoundError("未找到FASTA文件")
        
        all_sequences = []
        for file in fasta_files:
            file_path = os.path.join(data_dir, "raw", file)
            sequences = read_fasta(file_path)
            all_sequences.extend(sequences)
        
        # One-hot编码
        encoded_sequences = [one_hot_encode(seq) for seq in all_sequences]
        
        # 转换为NumPy数组
        X = np.array(encoded_sequences)
        
        # 生成示例标签（实际应用中应从数据中获取真实标签）
        y = np.random.randint(2, size=len(all_sequences))
        
        return X, y
    except Exception as e:
        print(f"数据准备过程中出错: {e}")
        raise

def train_model(config):
    """训练模型"""
    try:
        # 设置随机种子以确保可重复性
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # 加载数据
        data_dir = "data"
        X, y = prepare_data(data_dir, config)
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config["training"]["test_split"], random_state=42)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=config["training"]["validation_split"], random_state=42)
        
        # 构建模型
        model = build_hbv_cnn_model(input_shape=config["model"]["input_shape"])
        model.compile(
            optimizer=config["model"]["optimizer"], 
            loss=config["model"]["loss"], 
            metrics=[config["model"]["metrics"]]
        )
        
        # 确保模型保存目录存在
        model_save_dir = os.path.join("models", "saved_models")
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        
        # 设置回调函数
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(model_save_dir, "best_model.h5"),
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
            epochs=config["training"]["epochs"],
            batch_size=config["training"]["batch_size"],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
        )
        
        # 保存最终模型
        model.save(os.path.join(model_save_dir, "final_model.h5"))
        
        return history
    except Exception as e:
        print(f"模型训练过程中出错: {e}")
        raise