import tensorflow as tf
from deepfm6 import DeepFM
from tensorflow.keras.utils import plot_model

# 定义 CSV 文件的解析函数
def parse_line(line, num_dense=13, num_sparse=26):
    fields = tf.strings.split(line, ',')
    labels = tf.strings.to_number(fields[0], out_type=tf.float32)
    
    # 先将稠密特征中的空字符串填充为 '0'
    dense_features = fields[1:num_dense + 1]
    dense_features = tf.where(tf.equal(dense_features, ''), tf.fill(tf.shape(dense_features), '0'), dense_features)
    
    # 然后将稠密特征转换为浮点数
    dense_features = tf.strings.to_number(dense_features, out_type=tf.float32)
    
    # 对稀疏特征，处理缺失值，将其设为'-1'
    sparse_features = fields[num_dense + 1:]
    sparse_features = tf.where(tf.equal(sparse_features, ''), tf.fill(tf.shape(sparse_features), '-1'), sparse_features)
    
    return {'dense_inputs': dense_features, 'sparse_inputs': sparse_features}, labels

# 将字符串稀疏特征转换为哈希值的函数
def map_string_to_hash(features, label, num_buckets=100000):
    sparse_features = features['sparse_inputs']
    dense_features = features['dense_inputs']
    
    # 使用 tf.map_fn 对每个稀疏特征进行哈希
    hashed_sparse_features = tf.map_fn(lambda x: tf.strings.to_hash_bucket_fast(x, num_buckets=num_buckets), sparse_features, dtype=tf.int64)
    
    return {'dense_inputs': dense_features, 'sparse_inputs': hashed_sparse_features}, label

# 创建数据集的函数
def create_dataset(file_paths, batch_size, num_dense=13, num_sparse=26, num_buckets=100000):
    dataset = tf.data.TextLineDataset(file_paths).skip(1)
    dataset = dataset.map(lambda line: parse_line(line, num_dense, num_sparse))
    
    # 将稀疏特征转换为哈希值
    dataset = dataset.map(lambda features, label: map_string_to_hash(features, label, num_buckets))
    
    dataset = dataset.shuffle(100).batch(batch_size)
    return dataset

if __name__ == '__main__':
    # 数据集路径
    train_files = ["../examples/criteo_sample.txt"]
    val_files = ["../examples/criteo_sample.txt"]

    # 创建训练集和验证集
    train_dataset = create_dataset(train_files, batch_size=32)
    val_dataset = create_dataset(val_files, batch_size=32)

    # 定义模型参数
    sparse_feature_number = 100000  # 假设哈希桶数量为100000
    sparse_feature_dim = 10
    dense_feature_dim = 13
    sparse_num_field = 26
    layer_sizes = [128, 64, 32]

    # 定义模型
    model = DeepFM(sparse_feature_number, sparse_feature_dim, dense_feature_dim, sparse_num_field, layer_sizes)

    # 编译模型
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 训练模型
    model.fit(train_dataset, validation_data=val_dataset, epochs=10)

    # 绘制并保存模型结构图
    model_graph = model.build_graph(input_shape=(sparse_num_field, dense_feature_dim))
    plot_model(model_graph, to_file='deepfm_model.png', show_shapes=True, show_layer_names=True)

    # 保存模型
    model.summary()
    tf.saved_model.save(model, './model/deepfm_model_feat5_sparsedim9')

    # 绘制模型图像
    plot_model(model, to_file='./img/model_plot_feat5_sparsedim91.png', show_shapes=True, show_layer_names=True)
