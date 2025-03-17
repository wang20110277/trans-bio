import tensorflow as tf
from tensorflow.keras import layers, models

class TransformerBlock(layers.Layer):
    """Transformer 模块"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        # 多头自注意力机制
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # 前馈神经网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PositionalEncoding(layers.Layer):
    """位置编码"""
    def __init__(self, position, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, embed_dim)

    def get_angles(self, position, i, embed_dim):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(embed_dim, tf.float32))
        return position * angles

    def positional_encoding(self, position, embed_dim):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(embed_dim, dtype=tf.float32)[tf.newaxis, :],
            embed_dim=embed_dim,
        )
        # 对偶数索引应用正弦函数
        sines = tf.math.sin(angle_rads[:, 0::2])
        # 对奇数索引应用余弦函数
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], :]

def build_transformer_model(
    input_shape,
    num_heads=8,
    embed_dim=64,
    ff_dim=256,
    num_blocks=4,
    dropout_rate=0.1,
    num_classes=1,
):
    """构建 Transformer 模型"""
    inputs = layers.Input(shape=input_shape)

    # 嵌入层（将输入序列映射到高维空间）
    embedding = layers.Dense(embed_dim)(inputs)

    # 位置编码
    positional_encoding = PositionalEncoding(input_shape[0], embed_dim)
    x = positional_encoding(embedding)

    # 添加多个 Transformer 模块
    for _ in range(num_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)

    # 全局平均池化
    x = layers.GlobalAveragePooling1D()(x)

    # 全连接层
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    # 输出层
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

    # 构建模型
    model = models.Model(inputs, outputs)
    return model

if __name__ == "__main__":
    # 示例：构建 Transformer 模型
    input_shape = (100, 4)  # 输入形状（序列长度 x 4个核苷酸）
    model = build_transformer_model(input_shape)

    # 打印模型摘要
    model.summary()