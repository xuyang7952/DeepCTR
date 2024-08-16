import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import plot_model

# DNN部分
class DNN(layers.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim, dense_feature_dim, num_field, layer_sizes):
        super(DNN, self).__init__()
        sizes = [sparse_feature_dim * num_field] + layer_sizes + [1]
        self.mlp_layers = []

        for i in range(len(layer_sizes) + 1):
            self.mlp_layers.append(layers.Dense(
                units=sizes[i + 1],
                activation='relu' if i < len(layer_sizes) else None,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                    stddev=1.0 / tf.sqrt(float(sizes[i]))),
                name=f'dnn_dense_layer_{i}'
            ))

    def call(self, feat_embeddings):
        y_dnn = tf.reshape(feat_embeddings, [-1, feat_embeddings.shape[1] * feat_embeddings.shape[2]], name='reshape_feat_embeddings')
        for layer in self.mlp_layers:
            y_dnn = layer(y_dnn)
        return y_dnn

# FM部分
class FM(layers.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim, dense_feature_dim, sparse_num_field):
        super(FM, self).__init__()
        self.embedding_one = layers.Embedding(input_dim=sparse_feature_number, output_dim=1, name='sparse_embedding_one')
        self.embedding = layers.Embedding(input_dim=sparse_feature_number, output_dim=sparse_feature_dim, name='sparse_embedding')
        self.dense_w_one = self.add_weight(shape=(dense_feature_dim,), initializer='random_normal', trainable=True, name='dense_weight_one')
        self.dense_w = self.add_weight(shape=(1, dense_feature_dim, sparse_feature_dim), initializer='random_normal', trainable=True, name='dense_weight')

    def call(self, sparse_inputs, dense_inputs):
        sparse_inputs_concat = tf.concat(sparse_inputs, axis=1, name='sparse_inputs_concat')
        sparse_emb_one = self.embedding_one(sparse_inputs_concat)
        dense_emb_one = tf.multiply(dense_inputs, self.dense_w_one[None, :], name='dense_emb_one')
        dense_emb_one = tf.expand_dims(dense_emb_one, axis=2, name='dense_emb_one_expanded')
        y_first_order = tf.reduce_sum(sparse_emb_one, axis=1, name='sparse_first_order') + tf.reduce_sum(dense_emb_one, axis=1, name='dense_first_order')

        sparse_embeddings = self.embedding(sparse_inputs_concat,)
        dense_inputs_re = tf.expand_dims(dense_inputs, axis=2, name='dense_inputs_re')
        dense_embeddings = tf.multiply(dense_inputs_re, self.dense_w, name='dense_embeddings')
        feat_embeddings = tf.concat([sparse_embeddings, dense_embeddings], axis=1, name='feat_embeddings')

        summed_features_emb = tf.reduce_sum(feat_embeddings, axis=1, name='summed_features_emb')
        summed_features_emb_square = tf.square(summed_features_emb, name='summed_features_emb_square')
        squared_features_emb = tf.square(feat_embeddings, name='squared_features_emb')
        squared_sum_features_emb = tf.reduce_sum(squared_features_emb, axis=1, name='squared_sum_features_emb')

        y_second_order = 0.5 * tf.reduce_sum(summed_features_emb_square - squared_sum_features_emb, axis=1, keepdims=True, name='second_order')
        return y_first_order, y_second_order, feat_embeddings

# DeepFM模型部分
class DeepFM(Model):
    def __init__(self, sparse_feature_number, sparse_feature_dim, dense_feature_dim, sparse_num_field, layer_sizes):
        super(DeepFM, self).__init__()
        self.fm = FM(sparse_feature_number, sparse_feature_dim, dense_feature_dim, sparse_num_field)
        self.dnn = DNN(sparse_feature_number, sparse_feature_dim, dense_feature_dim, sparse_num_field, layer_sizes)
        self.bias = self.add_weight(shape=(1,), initializer='zeros', trainable=True, name='bias')

    def call(self, inputs):
        sparse_inputs, dense_inputs = inputs['sparse_inputs'], inputs['dense_inputs']
        y_first_order, y_second_order, feat_embeddings = self.fm(sparse_inputs, dense_inputs)
        y_dnn = self.dnn(feat_embeddings)
        predict = tf.nn.sigmoid(self.bias + y_first_order + y_second_order + y_dnn, name='prediction')
        return predict

# 模型构建和可视化
def build_deepfm_model(sparse_feature_number, sparse_feature_dim, dense_feature_dim, sparse_num_field, layer_sizes):
    sparse_inputs = [tf.keras.Input(shape=(1,), name=f'sparse_input_{i}') for i in range(sparse_num_field)]
    dense_inputs = tf.keras.Input(shape=(dense_feature_dim,), name='dense_inputs')
    model = DeepFM(sparse_feature_number, sparse_feature_dim, dense_feature_dim, sparse_num_field, layer_sizes)
    model_output = model({'sparse_inputs': sparse_inputs, 'dense_inputs': dense_inputs})
    return Model(inputs={'sparse_inputs': sparse_inputs, 'dense_inputs': dense_inputs}, outputs=model_output, name='DeepFM_Model')

# 参数定义
sparse_feature_number = 10000
sparse_feature_dim = 9
dense_feature_dim = 2
sparse_num_field = 3
layer_sizes = [128, 64, 32]

# 构建并显示模型
model = build_deepfm_model(sparse_feature_number, sparse_feature_dim, dense_feature_dim, sparse_num_field, layer_sizes)
model.summary()
plot_model(model, to_file='./img/deepfm_model_plot.png', show_shapes=True, show_layer_names=True)
