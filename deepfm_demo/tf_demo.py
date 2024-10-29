import tensorflow as tf
import numpy as np
import pandas as pd

# 生成合成数据
num_samples = 1000
input_data = np.random.rand(num_samples, 2)  # 两个输入特征
output_data = (input_data[:, 0] * 2 + input_data[:, 1] * 3 + np.random.normal(0, 0.1, num_samples)).reshape(-1, 1)

# 构建模型
inputs = tf.keras.Input(shape=(2,))
x = tf.keras.layers.Dense(16, activation="relu")(inputs)
outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="mse")

# 训练模型
model.fit(input_data, output_data, epochs=10, batch_size=32)

# 保存模型
model.save('./model/simplemodel',save_format='tf')

model.predict(input_data)
