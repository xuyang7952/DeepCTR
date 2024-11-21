import tensorflow as tf
import pandas as pd
import gc
import logging
from deepfm import DeepFMSimple  # 假设你已经定义了 DeepFMSimple 模型

logger = logging.getLogger(__name__)

def train_model(sparse_features, dense_features, df_names, target, feature_dict, conf):
    # 定义模型
    model = DeepFMSimple(sparse_features, dense_features, feature_dict, sparse_embdim=4, dnn_dropout=0.1, task='binary')
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['binary_crossentropy', 'accuracy', 'AUC'])

    # 使用 tf.data.Dataset 读取和预处理数据
    def process_features(chunk, target, feature_dict):
        # 这里定义你的预处理逻辑
        # 例如：转换类别特征为 one-hot 编码，标准化数值特征等
        # 假设你已经有了一个预处理函数 process_features
        train = process_features(chunk, target, feature_dict)
        return train[sparse_features + dense_features], train[target].values

    def parse_csv(line):
        record_defaults = [tf.float32] * len(df_names)
        parsed_line = tf.io.decode_csv(line, record_defaults=record_defaults)
        features = dict(zip(df_names, parsed_line))
        label = features.pop(target)
        return features, label

    def preprocess(features, label):
        train = process_features(features, label, feature_dict)
        return train

    for epoch_idx in range(conf['epochs']):
        dataset = tf.data.experimental.make_csv_dataset(
            conf['train_path'],
            batch_size=conf['batch_size'],
            column_names=df_names,
            select_columns=sparse_features + dense_features + [target],
            shuffle_buffer_size=None,  # 不进行 shuffle
            num_epochs=1,
            header=False,
            field_delim=','
        )

        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.cache().batch(conf['batch_size']).prefetch(tf.data.AUTOTUNE)

        history = model.fit(dataset, epochs=conf['chunk_epoch'], verbose=2, validation_split=0.1)
        logger.info(f"Epoch {epoch_idx} training history: {history.history}")

    # 保存模型
    del_file(conf['model_path'])
    model.save(conf['model_path'], save_format='tf')

def del_file(path):
    import os
    if os.path.exists(path):
        os.remove(path)

# 示例配置
conf = {
    'epochs': 5,
    'train_path': 'path_to_train_data.csv',
    'chunksize': 1000000,
    'batch_size': 32,
    'chunk_epoch': 1,
    'model_path': 'path_to_save_model'
}

# 示例特征和目标
sparse_features = ['feature1', 'feature2']
dense_features = ['feature3', 'feature4']
df_names = ['feature1', 'feature2', 'feature3', 'feature4', 'target']
target = 'target'
feature_dict = {}  # 假设你已经定义了 feature_dict

# 训练模型
train_model(sparse_features, dense_features, df_names, target, feature_dict, conf)


tf.data.Dataset