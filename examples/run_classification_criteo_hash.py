import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat,get_feature_names
import tensorflow as tf 

if __name__ == "__main__":
    data = pd.read_csv('./criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 2)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.do simple Transformation for dense features
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.set hashing space for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=1000,embedding_dim=4, use_hash=True, dtype='string')  # since the input is string
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                          for feat in dense_features]

    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns, )

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2, random_state=2020)

    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}


    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns,dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
    
    # 输出模型文件 
    # 指定保存模型的路径  
    export_dir = './saved_model/deepfm_criteo_hash' 
    
    # 保存模型  
    tf.saved_model.save(model, export_dir)

    # 打印模型结构
    print("model", model.summary())
    
    from tensorflow.keras.utils import plot_model  
  
    # 假设你已经有了一个模型 model  
    plot_model(model, to_file='./img/model_plot_deepfm_criteo_hash.png', show_shapes=True, show_layer_names=True)
    
    print("*"*30+"end"+"*"*30)
