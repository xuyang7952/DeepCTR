import tensorflow as tf
from collections import namedtuple, OrderedDict
from copy import copy
from itertools import chain

from tensorflow.python.keras.initializers import RandomNormal, Zeros
from tensorflow.python.keras.layers import Input, Lambda

from .inputs import create_embedding_matrix, embedding_lookup, get_dense_input, varlen_embedding_lookup, \
    get_varlen_pooling_list, mergeDict
from .layers import Linear
from .layers.utils import concat_func

DEFAULT_GROUP_NAME = "default_group"


class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'vocabulary_path', 'dtype', 'embeddings_initializer',
                             'embedding_name',
                             'group_name', 'trainable'])):
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, vocabulary_path=None, dtype="int32", embeddings_initializer=None,
                embedding_name=None,
                group_name=DEFAULT_GROUP_NAME, trainable=True):

        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if embeddings_initializer is None:
            embeddings_initializer = RandomNormal(mean=0.0, stddev=0.0001, seed=2020)

        if embedding_name is None:
            embedding_name = name

        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, vocabulary_path, dtype,
                                              embeddings_initializer,
                                              embedding_name, group_name, trainable)

    def __hash__(self):
        return self.name.__hash__()


class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
                                  ['sparsefeat', 'maxlen', 'combiner', 'length_name', 'weight_name', 'weight_norm'])):
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, combiner="mean", length_name=None, weight_name=None, weight_norm=True):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner, length_name, weight_name,
                                                    weight_norm)

    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def vocabulary_path(self):
        return self.sparsefeat.vocabulary_path

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embeddings_initializer(self):
        return self.sparsefeat.embeddings_initializer

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    @property
    def trainable(self):
        return self.sparsefeat.trainable

    def __hash__(self):
        return self.name.__hash__()


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype', 'transform_fn'])):
    """ Dense feature
    Args:
        name: feature name.
        dimension: dimension of the feature, default = 1.
        dtype: dtype of the feature, default="float32".
        transform_fn: If not `None` , a function that can be used to transform
        values of the feature.  the function takes the input Tensor as its
        argument, and returns the output Tensor.
        (e.g. lambda x: (x - 3.0) / 4.2).
    """
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32", transform_fn=None):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype, transform_fn)

    def __hash__(self):
        return self.name.__hash__()

    # def __eq__(self, other):
    #     if self.name == other.name:
    #         return True
    #     return False

    # def __repr__(self):
    #     return 'DenseFeat:'+self.name


def get_feature_names(feature_columns):
    features = build_input_features(feature_columns)
    return list(features.keys())


def build_input_features(feature_columns, prefix=''):
    input_features = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            input_features[fc.name] = Input(
                shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, DenseFeat):
            input_features[fc.name] = Input(
                shape=(fc.dimension,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, VarLenSparseFeat):
            input_features[fc.name] = Input(shape=(fc.maxlen,), name=prefix + fc.name,
                                            dtype=fc.dtype)
            if fc.weight_name is not None:
                input_features[fc.weight_name] = Input(shape=(fc.maxlen, 1), name=prefix + fc.weight_name,
                                                       dtype="float32")
            if fc.length_name is not None:
                input_features[fc.length_name] = Input((1,), name=prefix + fc.length_name, dtype='int32')

        else:
            raise TypeError("Invalid feature column type,got", type(fc))

    return input_features


def get_linear_logit(features, feature_columns, units=1, use_bias=False, seed=1024, prefix='linear',
                     l2_reg=0, sparse_feat_refine_weight=None):
    linear_feature_columns = copy(feature_columns)
    for i in range(len(linear_feature_columns)):
        if isinstance(linear_feature_columns[i], SparseFeat):
            linear_feature_columns[i] = linear_feature_columns[i]._replace(embedding_dim=1,
                                                                           embeddings_initializer=Zeros())
        if isinstance(linear_feature_columns[i], VarLenSparseFeat):
            linear_feature_columns[i] = linear_feature_columns[i]._replace(
                sparsefeat=linear_feature_columns[i].sparsefeat._replace(embedding_dim=1,
                                                                         embeddings_initializer=Zeros()))

    linear_emb_list = [input_from_feature_columns(features, linear_feature_columns, l2_reg, seed,
                                                  prefix=prefix + str(i))[0] for i in range(units)]
    _, dense_input_list = input_from_feature_columns(features, linear_feature_columns, l2_reg, seed, prefix=prefix)

    linear_logit_list = []
    for i in range(units):

        if len(linear_emb_list[i]) > 0 and len(dense_input_list) > 0:
            sparse_input = concat_func(linear_emb_list[i])
            dense_input = concat_func(dense_input_list)
            if sparse_feat_refine_weight is not None:
                sparse_input = Lambda(lambda x: x[0] * tf.expand_dims(x[1], axis=1))(
                    [sparse_input, sparse_feat_refine_weight])
            linear_logit = Linear(l2_reg, mode=2, use_bias=use_bias, seed=seed)([sparse_input, dense_input])
        elif len(linear_emb_list[i]) > 0:
            sparse_input = concat_func(linear_emb_list[i])
            if sparse_feat_refine_weight is not None:
                sparse_input = Lambda(lambda x: x[0] * tf.expand_dims(x[1], axis=1))(
                    [sparse_input, sparse_feat_refine_weight])
            linear_logit = Linear(l2_reg, mode=0, use_bias=use_bias, seed=seed)(sparse_input)
        elif len(dense_input_list) > 0:
            dense_input = concat_func(dense_input_list)
            linear_logit = Linear(l2_reg, mode=1, use_bias=use_bias, seed=seed)(dense_input)
        else:   #empty feature_columns
            return Lambda(lambda x: tf.constant([[0.0]]))(list(features.values())[0])
        linear_logit_list.append(linear_logit)

    return concat_func(linear_logit_list)


def input_from_feature_columns(features, feature_columns, l2_reg, seed, prefix='', seq_mask_zero=True,
                               support_dense=True, support_group=False):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

    embedding_matrix_dict = create_embedding_matrix(feature_columns, l2_reg, seed, prefix=prefix,
                                                    seq_mask_zero=seq_mask_zero)
    group_sparse_embedding_dict = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns)
    dense_value_list = get_dense_input(features, feature_columns)
    if not support_dense and len(dense_value_list) > 0:
        raise ValueError("DenseFeat is not supported in dnn_feature_columns")

    sequence_embed_dict = varlen_embedding_lookup(embedding_matrix_dict, features, varlen_sparse_feature_columns)
    group_varlen_sparse_embedding_dict = get_varlen_pooling_list(sequence_embed_dict, features,
                                                                 varlen_sparse_feature_columns)
    group_embedding_dict = mergeDict(group_sparse_embedding_dict, group_varlen_sparse_embedding_dict)
    if not support_group:
        group_embedding_dict = list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict, dense_value_list


def input_from_feature_columns2(features, feature_columns, l2_reg, seed, prefix='', seq_mask_zero=True,
                               support_dense=True, support_group=False):
    """
    构建模型输入张量。

    本函数根据输入的特征列配置，将原始特征数据处理成深度学习模型可以接受的输入张量形式。
    它主要处理了稀疏特征、变长稀疏特征和稠密特征，并根据模型配置进行相应的组装。

    参数:
    - features: 原始特征数据，通常是一个字典，键为特征名，值为特征值张量。
    - feature_columns: 特征列配置列表，定义了模型需要哪些特征及其类型。
    - l2_reg: L2正则化系数，用于控制模型中嵌入向量的复杂度。
    - seed: 随机种子，用于初始化模型中的随机数生成。
    - prefix: 前缀字符串，用于命名模型中的变量。
    - seq_mask_zero: 是否将序列特征中的mask值置为0，这对于某些模型的计算是必要的。
    - support_dense: 是否支持稠密特征。如果为False且存在稠密特征，则会抛出错误。
    - support_group: 是否支持将特征分组。如果为False，则不分组。

    返回:
    - group_embedding_dict: 处理后的稀疏特征和变长稀疏特征的嵌入字典（或列表，取决于support_group参数）。
    - dense_value_list: 处理后的稠密特征值列表。如果模型不支持稠密特征，则为空列表。
    """

    # 过滤出稀疏特征列
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    # 过滤出变长稀疏特征列
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

    # 创建所有特征的嵌入矩阵字典,这里就是基本就是稀疏特征的embedding
    embedding_matrix_dict = create_embedding_matrix(feature_columns, l2_reg, seed, prefix=prefix,
                                                    seq_mask_zero=seq_mask_zero)
    # 稀疏特征lookup值，常用于将字符串或整数类别的特征映射到连续的整数索引
    group_sparse_embedding_dict = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns)
    # 获取稠密特征输入列表
    dense_value_list = get_dense_input(features, feature_columns)
    # 如果模型不支持稠密特征且存在稠密特征，抛出错误
    if not support_dense and len(dense_value_list) > 0:
        raise ValueError("DenseFeat is not supported in dnn_feature_columns")

    # 查找并组装变长稀疏特征的嵌入字典
    sequence_embed_dict = varlen_embedding_lookup(embedding_matrix_dict, features, varlen_sparse_feature_columns)
    # 获取变长稀疏特征的池化列表
    group_varlen_sparse_embedding_dict = get_varlen_pooling_list(sequence_embed_dict, features,
                                                                 varlen_sparse_feature_columns)
    # 合并稀疏特征和变长稀疏特征的嵌入字典,deepfm中只有group_sparse_embedding_dict
    group_embedding_dict = mergeDict(group_sparse_embedding_dict, group_varlen_sparse_embedding_dict)
    # 如果模型不支持分组，将嵌入字典转换为列表
    if not support_group:
        group_embedding_dict = list(chain.from_iterable(group_embedding_dict.values()))

    return group_embedding_dict, dense_value_list