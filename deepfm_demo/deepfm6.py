import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import plot_model

# DNN模块类
class DNN(layers.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, num_field, layer_sizes):
        super(DNN, self).__init__()
        # 初始化网络的参数
        self.sparse_feature_number = sparse_feature_number  # 稀疏特征的数量
        self.sparse_feature_dim = sparse_feature_dim  # 稀疏特征的维度
        self.dense_feature_dim = dense_feature_dim  # 密集特征的维度
        self.num_field = num_field  # 特征字段的数量
        self.layer_sizes = layer_sizes  # MLP网络中隐藏层的节点数列表

        # 构建MLP网络，size为[sparse_feature_dim * num_field, layer_sizes[0], ..., 1]
        sizes = [sparse_feature_dim * num_field] + self.layer_sizes + [1]
        # 创建一个空列表，用于存储MLP网络的每一层
        self.mlp_layers = []
        # 遍历隐藏层大小列表，创建每一层的神经网络层
        for i in range(len(layer_sizes) + 1):
            # 添加一个全连接层到mlp_layers列表中
            self.mlp_layers.append(layers.Dense(
                units=sizes[i + 1],  # 输出维度
                activation='relu' if i < len(layer_sizes) else None,  # 激活函数，除了最后一层都使用ReLU
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                    stddev=1.0 / tf.sqrt(float(sizes[i]))),  # 使用的标准差是基于当前层的输入维度来计算的
                name=f'dense_layer_{i}'  # 层的名称
            ))

    # 前向传播函数
    def call(self, feat_embeddings):
        # 将特征嵌入reshape为二维张量，以便作为全连接层的输入，这里num_field为特征字段的数量,稀疏特征和稠密特征数量之和
        y_dnn = tf.reshape(feat_embeddings, [-1, self.num_field * self.sparse_feature_dim], name='reshape_feat_embeddings')

        # 通过所有的全连接层（MLP层）进行前向传播
        for i, layer in enumerate(self.mlp_layers):
            y_dnn = layer(y_dnn)
            # 为每个MLP层的输出创建一个命名，以便在后续使用
            y_dnn = tf.identity(y_dnn, name=f'dnn_output_{i}')

        # 返回全连接层的最终输出
        return y_dnn

    def build_graph(self, input_shape):
        feat_embeddings = tf.keras.Input(shape=(self.num_field, self.sparse_feature_dim), name='feat_embeddings')
        return Model(inputs=feat_embeddings, outputs=self.call(feat_embeddings), name='DNN_Model')


# FM模块类
class FM(layers.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field):
        # 初始化函数，用于初始化模型的参数和层
        super(FM, self).__init__()
        # 稀疏特征的数量
        self.sparse_feature_number = sparse_feature_number
        # 稀疏特征的维度
        self.sparse_feature_dim = sparse_feature_dim
        # 密集特征的维度
        self.dense_feature_dim = dense_feature_dim
        # 密集特征嵌入的维度，这里设置为稀疏特征的维度
        self.dense_emb_dim = self.sparse_feature_dim
        # 稀疏特征字段的数量
        self.sparse_num_field = sparse_num_field
        # 初始化参数值
        self.init_value_ = 0.1

        # 定义稀疏特征嵌入层，用于一阶特征交互
        self.embedding_one = layers.Embedding(
            input_dim=sparse_feature_number,
            output_dim=1,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(
                mean=0.0, stddev=self.init_value_ / tf.sqrt(float(self.sparse_feature_dim))),
            name='sparse_embedding_one'
        )

        # 定义稀疏特征嵌入层，用于二阶特征交互
        self.embedding = layers.Embedding(
            input_dim=self.sparse_feature_number,
            output_dim=self.sparse_feature_dim,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(
                mean=0.0, stddev=self.init_value_ / tf.sqrt(float(self.sparse_feature_dim))),
            name='sparse_embedding'
        )

        # 定义密集特征的权重，用于一阶特征交互
        self.dense_w_one = self.add_weight(
            shape=(self.dense_feature_dim,),
            initializer=tf.keras.initializers.TruncatedNormal(
                mean=0.0, stddev=self.init_value_ / tf.sqrt(float(self.sparse_feature_dim))),
            trainable=True,
            name='dense_weight_one'
        )

        # 定义密集特征的权重，用于二阶特征交互，连续特征也添加
        self.dense_w = self.add_weight(
            shape=(1, self.dense_feature_dim, self.dense_emb_dim),
            initializer=tf.keras.initializers.TruncatedNormal(
                mean=0.0, stddev=self.init_value_ / tf.sqrt(float(self.sparse_feature_dim))),
            trainable=True,
            name='dense_weight'
        )

    # 前向传播函数
    def call(self, sparse_inputs, dense_inputs):
        # -------------------- 一阶特征交互部分  --------------------
        # 获取稀疏特征输入
        sparse_inputs_concat = tf.concat(sparse_inputs, axis=1, name='sparse_inputs_concat')
        # [输入：[batch_size, sparse_num_field],输出：[batch_size, sparse_num_field, 1]]
        # 对稀疏特征进行嵌入，使其具有一个隐层维度的表示
        sparse_emb_one = self.embedding_one(sparse_inputs_concat)

        # -------------------- 稠密特征交互部分  --------------------
        # 对稠密特征进行加权，[输入：[batch_size, dense_feature_dim],输出：[batch_size, dense_feature_dim, 1]]
        dense_emb_one = tf.multiply(dense_inputs, self.dense_w_one[None, :], name='dense_emb_one')
        # 稠密特征需要扩展添加个维度，稀疏特征做了一个Embedding为1的操作，为后续的reduce_sum，保持相同的维度数
        # [输入：[batch_size, dense_feature_dim],输出：[batch_size, dense_feature_dim, 1]]
        dense_emb_one = tf.expand_dims(dense_emb_one, axis=2, name='dense_emb_one_expanded')
        # 计算一阶特征交互输出
        y_first_order = tf.reduce_sum(sparse_emb_one, axis=1, name='sparse_first_order') + \
                        tf.reduce_sum(dense_emb_one, axis=1, name='dense_first_order')

        # -------------------- 二阶特征交互部分  --------------------
        # 对稀疏特征进行嵌入，使其具有隐层维度的表示
        sparse_embeddings = self.embedding(sparse_inputs_concat)
        # 重构稠密特征，以便与嵌入后的稀疏特征进行运算
        dense_inputs_re = tf.expand_dims(dense_inputs, axis=2, name='dense_inputs_re')
        # 对稠密特征进行嵌入，使其具有隐层维度的表示
        dense_embeddings = tf.multiply(dense_inputs_re, self.dense_w, name='dense_embeddings')

        # 将稀疏和稠密特征嵌入拼接在一起
        feat_embeddings = tf.concat([sparse_embeddings, dense_embeddings], axis=1, name='feat_embeddings')

        # 计算所有特征嵌入之和
        summed_features_emb = tf.reduce_sum(feat_embeddings, axis=1, name='summed_features_emb')
        # 计算特征嵌入之和的平方
        summed_features_emb_square = tf.square(summed_features_emb, name='summed_features_emb_square')

        # 计算每个特征嵌入的平方和
        squared_features_emb = tf.square(feat_embeddings, name='squared_features_emb')
        squared_sum_features_emb = tf.reduce_sum(squared_features_emb, axis=1, name='squared_sum_features_emb')

        # 计算二阶特征交互输出
        y_second_order = 0.5 * tf.reduce_sum(summed_features_emb_square - squared_sum_features_emb, axis=1, keepdims=True, name='second_order')

        # 返回一阶特征交互输出，二阶特征交互输出和所有特征的嵌入表示
        return y_first_order, y_second_order, feat_embeddings

def build_graph(self, input_shape):
    """
    构建模型的计算图。

    该方法主要用于构建整个模型的输入层，包括稀疏输入和密集输入，并将这些输入连接到模型中。

    参数:
    - input_shape: 输入形状的参数，该参数在本模型中未直接使用，但保留以兼容其他可能需要该参数的模型。

    返回:
    - 返回构建好的Keras模型。
    """
    # 创建稀疏输入层列表，每个输入层对应一个稀疏特征，名称根据特征索引命名
    sparse_inputs = [tf.keras.Input(shape=(1,), name=f'sparse_input_{i}') for i in range(self.sparse_num_field)]
    # 创建密集输入层，形状为密集特征维度，名称为'dense_inputs'
    dense_inputs = tf.keras.Input(shape=(self.dense_feature_dim,), name='dense_inputs')
    # 构建并返回模型，输入包括稀疏输入和密集输入，输出通过调用模型的call方法计算得到，模型名称为'FM_Model'
    return Model(inputs=[sparse_inputs, dense_inputs], outputs=self.call(sparse_inputs, dense_inputs), name='FM_Model')


# DeepFM模型类
class DeepFM(Model):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                dense_feature_dim, sparse_num_field, layer_sizes):
        """
        初始化DeepFM模型的参数。

        参数:
        - sparse_feature_number: 稀疏特征的数量。
        - sparse_feature_dim: 稀疏特征的维度。
        - dense_feature_dim: 密集特征的维度。
        - sparse_num_field: 稀疏特征字段的数量。
        - layer_sizes: DNN层的尺寸列表。

        说明:
        - 初始化包括FM（因子分解机）和DNN（深度神经网络）模型，以及偏置项。
        - FM模型用于处理特征的二阶组合，而DNN模型用于学习更高阶的特征组合。
        - 偏置项是模型的一个可训练参数，用于提升模型的灵活性和拟合能力。
        """
        super(DeepFM, self).__init__()
        self.fm = FM(sparse_feature_number, sparse_feature_dim,
                    dense_feature_dim, sparse_num_field)
        self.dnn = DNN(sparse_feature_number, sparse_feature_dim,
                    dense_feature_dim, dense_feature_dim + sparse_num_field,
                    layer_sizes)
        self.bias = self.add_weight(shape=(1,), initializer='zeros', trainable=True, name='bias')

    def call(self, inputs):
        """
        调用模型的主要入口，用于计算输入数据的预测值。

        该函数首先从输入中分离出稀疏和密集特征，然后通过FM（Factorization Machine）层计算出一阶、二阶效应，
        以及特征嵌入值。随后，通过DNN（Deep Neural Network）层对特征嵌入值进行进一步加工。最后，将FM的一阶、
        二阶效应以及DNN的输出相加，并经过sigmoid激活函数得到最终的预测值。

        参数:
        - inputs: 字典类型，包含稀疏和密集输入特征。

        返回值:
        - predict: 模型预测值，经过sigmoid函数处理。
        """

        # 从输入数据中分离出稀疏和密集特征
        sparse_inputs, dense_inputs = inputs['sparse_inputs'], inputs['dense_inputs']

        # 通过FM层计算一阶、二阶效应以及特征嵌入值
        y_first_order, y_second_order, feat_embeddings = self.fm(sparse_inputs, dense_inputs)

        # 通过DNN层对特征嵌入值进行进一步加工
        y_dnn = self.dnn(feat_embeddings)

        # 将FM的一阶、二阶效应以及DNN的输出相加，并经过sigmoid激活函数得到最终的预测值
        predict = tf.nn.sigmoid(self.bias + y_first_order + y_second_order + y_dnn, name='prediction')

        return predict

    def build_graph(self, input_shape):
        """
        构建DeepFM模型的计算图。

        该函数负责根据输入的形状创建模型的输入层，并连接这些输入层以构建完整的模型。
        模型的输入分为稀疏输入和密集输入两部分，其中稀疏输入为多个独立的稀疏特征，
        而密集输入为一个密集向量。通过这种方式，模型能够处理不同类型的数据输入。

        参数:
        input_shape: 元组，模型输入的形状，第一个元素为稀疏特征的数量，第二个元素为密集特征的维度。

        返回:
        tf.keras.Model: 构建的DeepFM模型，该模型的输入包括稀疏输入和密集输入，输出为模型的预测结果。
        """
        # 创建稀疏输入层列表，每个稀疏输入层的形状为(1,), 根据输入形状中的稀疏特征数量
        sparse_inputs = [tf.keras.Input(shape=(1,), name=f'sparse_input_{i}') for i in range(input_shape[0])]
        # 创建密集输入层，其形状根据输入形状中的密集特征维度
        dense_inputs = tf.keras.Input(shape=(input_shape[1],), name='dense_inputs')
        # 构建并返回DeepFM模型，输入包括稀疏输入和密集输入，输出为模型的预测结果
        return Model(inputs={'sparse_inputs': sparse_inputs, 'dense_inputs': dense_inputs},
                     outputs=self.call({'sparse_inputs': sparse_inputs, 'dense_inputs': dense_inputs}),
                     name='DeepFM_Model')
    
    
    
if __name__ == '__main__':
    print("*" * 30 + "start" + "*" * 30)
    # 创建DeepFM模型实例
    model = DeepFM(sparse_feature_number=10000, 
                sparse_feature_dim=9, 
                dense_feature_dim=2, 
                sparse_num_field=5, 
                layer_sizes=[128, 64, 32])

    # 编译模型，指定优化器，损失函数和评价指标
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 模拟稀疏输入数据
    sparse_inputs = [tf.random.uniform(shape=(64, 1), maxval=1000, dtype=tf.int32, name=f'sparse_input_{i}') for i in range(5)]
    dense_inputs = tf.random.uniform(shape=(64, 2), name='dense_inputs')

    # 创建输入数据的字典
    inputs = {'sparse_inputs': sparse_inputs, 'dense_inputs': dense_inputs}

    # 训练模型
    model.fit(x=inputs, y=tf.random.uniform(shape=(64, 1), maxval=2, dtype=tf.int32), epochs=10)


    # 构建并打印模型详细结构
    # model.build_graph([5, 2]).summary()
    model.summary()
    # model.save('./model/deepfm_model_feat5_sparsedim9',save_format='tf')
    tf.saved_model.save(model,'./model/deepfm_model_feat5_sparsedim9')

    plot_model(model, to_file='./img/model_plot_feat5_sparsedim9.png', show_shapes=True, show_layer_names=True)

    print("*" * 30 + "end" + "*" * 30)
