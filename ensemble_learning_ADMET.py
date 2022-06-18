"""集成模型预测"""
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import xgboost
from tensorflow.keras import layers, Model, Sequential
from sklearn.model_selection import train_test_split


def min_max(x):
    print('min:', np.min(x), 'max', np.max(x))
    if np.std(x) == 0:  # 避免0除
        return x
    return (x - np.mean(x)) / np.std(x)


"""read files"""
print('reading files...')
molecular = pd.read_excel('Molecular_Descriptor.xlsx', 0)
ER = pd.read_excel('ADMET.xlsx', 0)
Q3data = pd.merge(molecular, ER, on='SMILES')
print(Q3data)
test_admet = pd.read_excel('./ADMET.xlsx',1)
test_molecular = pd.read_excel('Molecular_Descriptor.xlsx', 1)
test = pd.merge(test_admet, test_molecular, on='SMILES')
test_res = pd.DataFrame([])
test_res['SMILES'] = test['SMILES']
print(test)

mole_columns = molecular.columns
X_all = Q3data[mole_columns[1:-1]]
print(list(X_all.columns))
# 所有自变量归一化处理
STD = False
if STD:
    for name in list(X_all.columns):
        d = min_max(X_all[name])
        Q3data = Q3data.drop(name, axis=1)
        Q3data[name] = d
print(Q3data)
"""测试集训练集划分"""
train, val = train_test_split(Q3data, test_size=0.2)


"""神经网络集成"""
from sklearn.metrics import accuracy_score
#每个MDET指标都有一个list指标

class EmbedingModel(tf.keras.Model):
    def __init__(self):
        super(EmbedingModel, self).__init__()
        self.fc1 = layers.Dense(15,activation='relu')
        self.fc2 = layers.Dense(2,activation='softmax')
        self.con = layers.Concatenate(axis=1)
        self.fc3 = layers.Dense(2,activation='softmax')

    def call(self, inputs):
        x1 = self.fc1(inputs[1])
        x1 = self.fc2(x1)
        x = self.con([inputs[0],x1])
        x = self.fc3(x)
        return x

def embed_model(train,val,test,num,name,cha_sel):
    loss = pd.DataFrame([])
    y_1 = tf.one_hot(train[name], depth=2)
    y_1_val = tf.one_hot(val[name], depth=2)
    X_train = train[cha_sel]
    X_val = val[cha_sel]
    """xgboost"""
    model_xg = xgboost.XGBClassifier().fit(X_train, train[name])

    """Lasso"""
    from sklearn.linear_model import Lasso
    lr = Lasso(alpha=0.1).fit(X_train, train[name])

    """随机森林"""
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1).fit(X_train, train[name])

    three_model_train = pd.DataFrame([])
    three_model_train['xg'] = model_xg.predict(X_train)
    three_model_train['lr'] = lr.predict(X_train)
    print('lasso Predict',lr.predict(X_train),"\n",np.round_(lr.predict(X_train)))
    three_model_train['rf'] = forest.predict(X_train)
    three_model_val = pd.DataFrame([])
    three_model_val['xg'] = model_xg.predict(X_val)
    three_model_val['lr'] = lr.predict(X_val)
    three_model_val['rf'] = forest.predict(X_val)
    loss['xg'] = [accuracy_score(model_xg.predict(X_train), train[name]), accuracy_score(model_xg.predict(X_val), val[name])]
    loss['lr'] = [accuracy_score(np.round_(lr.predict(X_train)), train[name]), accuracy_score(np.round_(lr.predict(X_val)), val[name])]
    loss['rf'] = [accuracy_score(forest.predict(X_train), train[name]), accuracy_score(forest.predict(X_val), val[name])]
    test_res['xg_'+name] = model_xg.predict(test[cha_sel])
    test_res['lr_' + name] = lr.predict(test[cha_sel])
    test_res['rf_' + name] = forest.predict(test[cha_sel])

    model_ES_1 = EmbedingModel()
    model_ES_1.build(input_shape=[(None,3),(None,num)])
    model_ES_1.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate = 2e-2),
                  metrics=['accuracy'])
    model_ES_1.summary()
    #转化为tensor
    three_model_train = tf.reshape(three_model_train,shape=(-1,3))
    X_train = tf.reshape(X_train,shape=(-1,num))
    three_model_val = tf.reshape(three_model_val,shape=(-1,3))
    X_val= tf.reshape(X_val,shape=(-1,num))
    #模型拟合
    model_ES_1.fit(x=[three_model_train,X_train ], y=y_1, epochs=250, batch_size=512,
                   verbose=1,
                   validation_data=([three_model_val,X_val ], y_1_val))
    score_train = model_ES_1.evaluate([three_model_train,X_train ], y_1, verbose=1)
    score_val = model_ES_1.evaluate([three_model_val,X_val], y_1_val, verbose=1)
    other_classfier_res = tf.reshape(test_res[['xg_'+name,'lr_' + name,'rf_' + name]],shape=(-1,3))
    y_test = model_ES_1([other_classfier_res,tf.reshape(test[cha_sel],shape=(-1,num))]).numpy()
    print(y_test)
    res_te = []
    for i in y_test:
        res_te.append(tf.argmax(i).numpy())
    test_res['ens_' + name] = res_te
    test_res.to_csv('./Q3/test_res_'+name+'.csv')
    loss['ourmethod'] = [score_train[1],score_val[1]]
    loss.to_csv('./Q3/ACC对比_'+name+'.csv')
    print('training loss:', score_train[0], 'train acc:', score_train[1])
    print('valid loss:', score_val[0], " valid acc:", score_val[1])

    return model_ES_1


name_list = ['Caco-2', 'CYP3A4', 'hERG', 'HOB', 'MN']
varriables = pd.read_csv('./Q3/变量选择.csv')
name_ = 'hERG'
mdet1_var = varriables[name_].to_list()
print(mdet1_var)
embed_model(train,val,test,20,name_,mdet1_var)