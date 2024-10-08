import tensorflow as tf
import yaml
import os
from deepctr.models import DeepFM

# 读取配置文件
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# 数据读取和预处理
def read_data(data_paths, batch_size, buffer_size):
    # 创建一个包含所有文件路径的列表
    filenames = tf.data.Dataset.from_tensor_slices(data_paths)
    
    # 从每个文件中读取数据
    def load_file(filename):
        # 创建一个文本文件数据集
        dataset = tf.data.TextLineDataset(filename).skip(1)  # 跳过第一行标题
        # 解析每一行数据
        def parse_csv(line):
            # 定义特征索引字典，其中键为特征索引，值为特征类型（'sparse' 或 'dense'）
            feature_indices = {
                1: 'sparse', 2: 'sparse', 3: 'dense', 4: 'sparse', 5: 'dense', 6: 'dense'
                # ... 其他特征索引
            }
            
            # 解析CSV行
            record_defaults = [[0]] * len(feature_indices)
            parsed_line = tf.io.decode_csv(line, record_defaults=record_defaults)

            # 根据索引字典分离特征
            sparse_features = []
            dense_features = []
            for index, feature_type in feature_indices.items():
                if feature_type == 'sparse':
                    sparse_features.append(parsed_line[index])
                else:
                    dense_features.append(parsed_line[index])

            # 分离特征和标签
            label = parsed_line[0]

            return {'sparse_inputs': sparse_features, 'dense_inputs': dense_features}, label

        # 应用解析函数
        dataset = dataset.map(parse_csv, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    # 从所有文件中读取数据
    dataset = filenames.flat_map(load_file)
    
    # 打乱数据
    dataset = dataset.shuffle(buffer_size)
    # 分批并预取数据
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# 加载配置文件
config_path = 'config.yaml'
config = load_config(config_path)

# 创建DeepFM模型实例
model = DeepFM(sparse_feature_number=config['sparse_feature_number'],
               sparse_feature_dim=config['sparse_feature_dim'],
               dense_feature_dim=config['dense_feature_dim'],
               sparse_num_field=config['sparse_num_field'],
               layer_sizes=config['layer_sizes'])

# 编译模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 读取数据
data_paths = config['data_paths']
batch_size = config['train_batch_size']
buffer_size = min(10000, len(data_paths))
dataset = read_data(data_paths, batch_size, buffer_size)

# 训练模型
model.fit(dataset, epochs=config['epochs'], verbose=1)

# 保存模型
model.save(config['model_save_path'], save_format='tf')