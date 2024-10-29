# tensorflow2.7 实现 mnist的demo案例，并保存模型
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# 加载MNIST数据集
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# 数据预处理
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 归一化
train_images, test_images = train_images / 255.0, test_images / 255.0

# 创建模型
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 输出层，使用softmax分类
])

# 编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# 训练模型
history = model.fit(train_images, train_labels, epochs=5,
                    validation_data=(test_images, test_labels))

# 评估模型
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# 保存整个模型
model.save('./model/minst',save_format='tf')

# 保存模型的方式也可以使用回调函数在训练过程中自动保存模型
# checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath='mnist_model_{epoch}.h5',
#     save_weights_only=False,
#     monitor='val_accuracy',
#     mode='max',
#     save_best_only=True)

# history = model.fit(train_images, train_labels, epochs=5,
#                     validation_data=(test_images, test_labels),
#                     callbacks=[checkpoint_callback])