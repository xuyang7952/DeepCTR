import tensorflow as tf  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Embedding, Flatten  
  
# 创建一个简单的模型  
# 假设我们的词汇表大小为10（即input_dim=10），每个词汇的Embedding向量大小为5  
model = Sequential([  
    Embedding(input_dim=10, output_dim=5, input_length=1),  # input_length=1意味着我们每次只输入一个索引  
    Flatten()  # 将Embedding层的输出展平，以便可以直接用于全连接层或其他层  
])  
  
# 编译模型（虽然这个示例不会进行训练，但编译是检查模型结构的好方法）  
model.compile(optimizer='adam', loss='mse')  # 这里使用均方误差作为损失函数，仅用于示例  
  
# 创建一个包含索引0的输入数据（形状为(batch_size, input_length)）  
# 在这个例子中，我们只有一个样本，所以batch_size=1  
input_data = tf.constant([[9]], dtype=tf.int32)  # 注意索引是整数类型  
  
# 使用模型进行预测（虽然这里只是简单地通过模型传递数据，没有进行训练）  
output = model.predict(input_data)  

# 打印出model的所有参数
print(model.summary())
# 打印出模型的embedding层参数
print(model.layers[0].get_weights())

# 打印输出以验证  
print("Output shape:", output.shape)  # 应该是(1, 5)，因为我们有一个样本，每个样本的Embedding向量大小为5  
print("Output:", output)  # 打印实际的Embedding向量值  
  
# 注意：由于Embedding层是随机初始化的，所以每次运行这段代码时，输出的Embedding向量值都会不同