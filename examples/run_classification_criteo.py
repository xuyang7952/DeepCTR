import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
import tensorflow as tf
from loguru import logger

if __name__ == "__main__":
    data = pd.read_csv('./criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 3)]
    dense_features = ['I' + str(i) for i in range(1, 3)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                                            for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    print(f"data.head:{data.head()}")
    for feat in sparse_features:
        print(f"data[{feat}].value_counts:{data[feat].value_counts()}")
        print(f"data[{feat}].vocabulary_size:{data[feat].max()+1}")
    train, test = train_test_split(data, test_size=0.1, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    # 打印测试集5个样本数据
    print(f"test_model_input:{test_model_input}")

    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=256)
    # 打印pred_ans结构
    print(f"pred_ans:{pred_ans}")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
    
    # 输出模型文件 
    # 指定保存模型的路径  
    export_dir = './saved_model/deepfm_criteo' 
    
    # 保存模型  
    tf.saved_model.save(model, export_dir)

    # 打印模型结构,embedding维度是4，C1这里是27个，对应的embedding总参数是108个，C2是92个，对应的参数就是
    print("model", model.summary())
    # 打印模型的参数，每一层的参数
    
    # 打印模型的稀疏特征的embeding层的embedding值
    print()
    
    
    from tensorflow.keras.utils import plot_model  
  
    # 假设你已经有了一个模型 model  
    plot_model(model, to_file='./img/model_plot_deepfm_criteo2_nofm.png', show_shapes=True, show_layer_names=True)
    
    print("*"*30+"end"+"*"*30)