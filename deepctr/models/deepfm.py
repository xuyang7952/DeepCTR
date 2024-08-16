# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen, weichenswc@163.com

Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)

"""

from itertools import chain

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense

from ..feature_column import build_input_features, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import FM
from ..layers.utils import concat_func, add_func, combined_dnn_input


def DeepFM(linear_feature_columns, dnn_feature_columns, fm_group=(DEFAULT_GROUP_NAME,), dnn_hidden_units=(256, 128, 64),
           l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
           dnn_activation='relu', dnn_use_bn=False, task='binary'):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by the linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by the deep part of the model.
    :param fm_group: list, group_name of features that will be used to do feature interactions.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """
    """
    linear part and dnn part特征是相同的，都是稀疏特征的Embedding值和连续特征的值
    [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)for i, feat in enumerate(sparse_features)] 
    + [DenseFeat(feat, 1, )for feat in dense_features]
    """ 
    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())
    
    # linear part ，会把稀疏特征的Embedding的维度替换为1，这里相当于LR模型
    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    # 获取稀疏特征embedding 和连续特征
    """
    这里的group_embedding_dict得到的值是分group的特征的embedding值，特征的group概念是FFM模型会用到的，这里统一用默认的'default_group'值代替，全部稀疏特征视作一个组；
    group_embedding_dict:defaultdict(<class 'list'>, {'default_group': [<KerasTensor: shape=(None, 1, 4) dtype=float32 (created by layer 'sparse_emb_C1')>, <KerasTensor: shape=(None, 1, 4) dtype=float32 (created by layer 'sparse_emb_C2')>]})
    dense_value_list则直接是连续特征的值
    dense_value_list:[<KerasTensor: shape=(None, 1) dtype=float32 (created by layer 'I1')>, <KerasTensor: shape=(None, 1) dtype=float32 (created by layer 'I2')>]
    """
    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                                        seed, support_group=True)
    print(f"group_embedding_dict:{group_embedding_dict}")
    print(f"dense_value_list:{dense_value_list}")

    # FM模型结构，fmgrou应该可以扩展到FFM模型，只要扩充为多个组
    fm_logit = add_func([FM()(concat_func(v, axis=1))
                         for k, v in group_embedding_dict.items() if k in fm_group])
    
    # DNN结构  稀疏特征的Embedding值和连续特征的值
    dnn_input = combined_dnn_input(list(chain.from_iterable(
        group_embedding_dict.values())), dense_value_list)
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    dnn_logit = Dense(1, use_bias=False)(dnn_output)

    # LR + FM二阶 + DNN
    final_logit = add_func([linear_logit, fm_logit, dnn_logit])
    # final_logit = add_func([linear_logit, dnn_logit])

    output = PredictionLayer(task)(final_logit)
    model = Model(inputs=inputs_list, outputs=output)
    return model
