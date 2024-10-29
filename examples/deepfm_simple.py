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
import tensorflow as tf

def DeepFMSimple(sparse_features, dense_features, feature_dict, sparse_embdim=4, dnn_hidden_units=(256, 128, 64), 
                 l2_reg_embedding=0.001, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu', 
                 dnn_use_bn=False, task='binary'):
    """Instantiates the DeepFM Network architecture with a single input tensor for simplified TF Serving format."""

    embeddings_initializer = RandomNormal(mean=0.0, stddev=0.0001, seed=1024)
    
    # Define a single input layer for all features
    input_dim = len(sparse_features) + len(dense_features)
    inputs = Input(shape=(input_dim,), name="inputs")
    
    # Split sparse and dense parts from the input tensor
    sparse_inputs = inputs[:, :len(sparse_features)]
    dense_inputs = inputs[:, len(sparse_features):]
    
    # Process sparse inputs: Apply embeddings to each sparse feature slice
    sparse_embeds = [Embedding(input_dim=feature_dict[feat]["sparse_size"], output_dim=sparse_embdim,
                               embeddings_initializer=embeddings_initializer, embeddings_regularizer=l2(l2_reg_embedding))(
                     tf.cast(sparse_inputs[:, i], tf.int32)) for i, feat in enumerate(sparse_features)]
    
    # Flatten embeddings
    sparse_embeds_flatten = [Flatten()(embed) for embed in sparse_embeds]

    # DNN input combines sparse embeddings and dense inputs directly
    dnn_input = Concatenate()(sparse_embeds_flatten + [dense_inputs])
    
    # DNN part
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    dnn_logit = Dense(1, use_bias=False)(dnn_output)

    # FM part
    fm_input = Reshape((len(sparse_embeds_flatten), sparse_embdim))(Concatenate()(sparse_embeds))
    fm_logit = FM()(fm_input)

    # Linear part
    sparse_embed1_flatten = [Flatten()(Embedding(input_dim=feature_dict[feat]["sparse_size"], output_dim=1, 
                               embeddings_initializer=Zeros(), embeddings_regularizer=l2(l2_reg_embedding))(
                               tf.cast(sparse_inputs[:, i], tf.int32))) for i, feat in enumerate(sparse_features)]
    linear_logit = Dense(1, activation=None)(Concatenate()(sparse_embed1_flatten + [dense_inputs]))

    # Combine outputs
    final_logit = dnn_logit + fm_logit + linear_logit
    output = PredictionLayer(task)(final_logit)

    # Define and return the model
    model = Model(inputs=inputs, outputs=output)
    return model
