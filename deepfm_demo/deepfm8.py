import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Add
from tensorflow.keras.regularizers import l2 # type: ignore
from itertools import chain

# 定义特征列相关的类和方法
class SparseFeat:
    def __init__(self, name, vocabulary_size, embedding_dim):
        self.name = name
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim

class DenseFeat:
    def __init__(self, name, dimension):
        self.name = name
        self.dimension = dimension

# 构建输入特征
def build_input_features(feature_columns):
    inputs = {}
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            inputs[fc.name] = Input(shape=(1,), name=fc.name, dtype=tf.int32)
        elif isinstance(fc, DenseFeat):
            inputs[fc.name] = Input(shape=(fc.dimension,), name=fc.name, dtype=tf.float32)
    return inputs

# 获取线性部分的logit
def get_linear_logit(features, feature_columns, seed, prefix, l2_reg):
    linear_part = []
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            embed = tf.keras.layers.Embedding(fc.vocabulary_size, 1, embeddings_regularizer=l2(l2_reg))(features[fc.name])
            linear_part.append(tf.reduce_sum(embed, axis=-1))
        elif isinstance(fc, DenseFeat):
            linear_part.append(features[fc.name])
    linear_logit = tf.keras.layers.Dense(1, use_bias=False, kernel_regularizer=l2(l2_reg), name=f'{prefix}_logit')(Concatenate()(linear_part))
    return linear_logit

# 处理输入特征
def input_from_feature_columns(features, feature_columns, l2_reg_embedding, seed, support_group):
    group_embedding_dict = {}
    dense_value_list = []

    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            embed = tf.keras.layers.Embedding(fc.vocabulary_size, fc.embedding_dim, embeddings_regularizer=l2(l2_reg_embedding))(features[fc.name])
            group_embedding_dict.setdefault(DEFAULT_GROUP_NAME, []).append(embed)
        elif isinstance(fc, DenseFeat):
            dense_value_list.append(features[fc.name])

    return group_embedding_dict, dense_value_list

# 深度神经网络部分
def DNN(hidden_units, activation, l2_reg, dropout_rate, use_bn, seed):
    def _DNN(x):
        for units in hidden_units:
            x = tf.keras.layers.Dense(units, activation=None, kernel_regularizer=l2(l2_reg))(x)
            if use_bn:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(activation)(x)
            x = tf.keras.layers.Dropout(dropout_rate, seed=seed)(x)
        return x
    return _DNN

# 最终模型
def DeepFM(linear_feature_columns, dnn_feature_columns, fm_group=(DEFAULT_GROUP_NAME,), dnn_hidden_units=(256, 128, 64),
           l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
           dnn_activation='relu', dnn_use_bn=False, task='binary'):
    features = build_input_features(linear_feature_columns + dnn_feature_columns)
    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear', l2_reg=l2_reg_linear)

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding, seed, support_group=True)

    fm_logit = Add()([tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x * x - tf.reduce_sum(x, axis=1, keepdims=True)**2 / 2, axis=1, keepdims=True))(Concatenate(axis=1)(v)) for k, v in group_embedding_dict.items() if k in fm_group])

    dnn_input = Concatenate()(list(chain.from_iterable(group_embedding_dict.values())) + dense_value_list)
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    dnn_logit = Dense(1, use_bias=False)(dnn_output)

    final_logit = Add()([linear_logit, fm_logit, dnn_logit])

    output = tf.keras.layers.Dense(1, activation='sigmoid' if task == 'binary' else None)(final_logit)
    model = Model(inputs=inputs_list, outputs=output)
    return model

# 默认组名
DEFAULT_GROUP_NAME = 'default_group'

# 示例使用
sparse_features = ['sparse1', 'sparse2']
dense_features = ['dense1', 'dense2']

sparse_feature_columns = [SparseFeat(feat, vocabulary_size=1000, embedding_dim=4) for feat in sparse_features]
dense_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]

model = DeepFM(sparse_feature_columns, dense_feature_columns, dnn_hidden_units=(256, 128, 64))
model.summary()