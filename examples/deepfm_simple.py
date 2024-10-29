# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen, weichenswc@163.com

Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)

"""

from itertools import chain

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import  Input, Dense, Concatenate, Embedding, Flatten,Reshape
from tensorflow.python.keras.initializers import RandomNormal, Zeros
from tensorflow.python.keras.regularizers import l2

from deepctr.feature_column import build_input_features, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.interaction import FM
from deepctr.layers.utils import concat_func, add_func, combined_dnn_input
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names


def DeepFMSimple(sparse_features,dense_features, feature_dict,sparse_embdim=4, dnn_hidden_units=(256, 128, 64),l2_reg_embedding=0.001, 
                 l2_reg_dnn=0, seed=1024, dnn_dropout=0,dnn_activation='relu', dnn_use_bn=False, task='binary'):
    """Instantiates the DeepFM Network architecture.
    """ 
    embeddings_initializer = RandomNormal(mean=0.0, stddev=0.0001, seed=1024)
    # 定义稀疏和密集特征的输入层
    sparse_inputs = {feat: Input(shape=(1,), name=feat, dtype='int32') for feat in sparse_features}
    dense_inputs = {feat: Input(shape=(1,), name=feat, dtype='float32') for feat in dense_features}
    
    # 稀疏特征经过嵌入层,embedding_dim=1
    sparse_embed1 = [Embedding(input_dim=feature_dict[feat]["sparse_size"], output_dim=1,embeddings_initializer=Zeros(),
                               embeddings_regularizer=l2(l2_reg_embedding))(sparse_inputs[feat])
                     for feat in sparse_features ]
    
    # 稀疏特征经过嵌入层，embedding_dim=sparse_embdim
    sparse_embeds = [Embedding(input_dim=feature_dict[feat]["sparse_size"], output_dim=sparse_embdim,embeddings_initializer=embeddings_initializer,
                               embeddings_regularizer=l2(l2_reg_embedding))(sparse_inputs[feat])
                     for feat in sparse_features ]
    # 展平嵌入结果
    sparse_embed1_flatten = [Flatten()(embed) for embed in sparse_embed1]
    sparse_embeds_flatten = [Flatten()(embed) for embed in sparse_embeds]

    # 将密集特征拼接为列表
    dense_values = list(dense_inputs.values())

    # 将嵌入的稀疏特征和密集特征拼接在一起
    dnn_input = Concatenate()(sparse_embeds_flatten + dense_values)

    # DNN部分
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    dnn_logit = Dense(1, use_bias=False)(dnn_output)

    # FM部分
    # FM部分，使用 Reshape 进行维度调整
    fm_input = Reshape((len(sparse_embeds_flatten), sparse_embdim))(Concatenate()(sparse_embeds))
    fm_logit = FM()(fm_input)

    # 线性部分
    linear_logit = Dense(1, activation=None)(Concatenate()(sparse_embed1_flatten + dense_values))

    # 合并输出层
    final_logit = dnn_logit + fm_logit + linear_logit
    output = PredictionLayer(task)(final_logit)

    # 构建模型
    model = Model(inputs=list(sparse_inputs.values()) + list(dense_inputs.values()), outputs=output)
    return model
