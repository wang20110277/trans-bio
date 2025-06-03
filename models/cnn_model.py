import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # 假设是二分类问题
    return model

def build_hbv_cnn_model(input_shape):
    model = models.Sequential()
    # 可能需要调整卷积层和池化层的参数
    model.add(layers.Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=3))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # 假设是二分类问题
    return model