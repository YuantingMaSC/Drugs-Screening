import pandas as pd
import numpy as np
import xgboost
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras import Sequential,Model,layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns


def min_max(x):
    print('min:',np.min(x),'max',np.max(x))
    if np.std(x) == 0:  #避免0除
        return x
    return (x-np.mean(x))/np.std(x)

lr = 0.08

class EmbedingModel(tf.keras.Model):
    def __init__(self):
        super(EmbedingModel, self).__init__()
        self.fc1 = layers.Dense(10,activation='relu')
        self.fc2 = layers.Dense(1,activation='relu')
        self.con = layers.Concatenate(axis=1)
        self.fc3 = layers.Dense(1)

    def call(self, inputs):
        x1 = self.fc1(inputs[1]) #[[7,8,7.5][,,,,,,,,,,]]
        x1 = self.fc2(x1) # 对20个分子描述符的降维
        x = self.con([x1,inputs[0]]) #[7,7,7,7]
        x = self.fc3(x)
        return x

"""read files"""
print('reading files...')
molecular = pd.read_excel('Molecular_Descriptor.xlsx',0)
ER = pd.read_excel('ERα_activity.xlsx',0)
Q1data = pd.merge(molecular,ER,on='SMILES')
charac_select = ['pIC50','MDEC-23','BCUTp-1h','ATSp4','ALogP','MLFER_A','minsssN','ATSp1','minHsOH','LipoaffinityIndex','mindO','ATSp5','C1SP2','ATSc4','C3SP2','maxsssCH','maxHBd','minHBint10','ATSc3','maxHsOH','AMR']
data_all = Q1data[charac_select]
print(data_all)


"""train_val_split"""
charac = ['MDEC-23','BCUTp-1h','ATSp4','ALogP','MLFER_A','minsssN','ATSp1','minHsOH','LipoaffinityIndex','mindO','ATSp5','C1SP2','ATSc4','C3SP2','maxsssCH','maxHBd','minHBint10','ATSc3','maxHsOH','AMR']
train,val = train_test_split(data_all, test_size=0.20)
train = train.reset_index()
val = val.reset_index()  #重置索引
y_2_train = train['pIC50']
X_train = train[charac]
y_2_val = val['pIC50']
X_val = val[charac]

Loss_res = []

from sklearn.metrics import mean_squared_error
"""xgboost"""
model_xg = xgboost.XGBRegressor().fit(X_train , y_2_train)
y_xg = model_xg.predict(X_train)
y_xg_embed = model_xg.predict(X_val)
Loss_res.append([mean_squared_error(y_xg,y_2_train),mean_squared_error(y_xg_embed,y_2_val)])


"""lasso"""
model_lasso =  Lasso(alpha = 0.1)
model_lasso.fit(X_train,y_2_train)
y_lasso = model_lasso.predict(X_train)
y_lasso_val = model_lasso.predict(X_val)
Loss_res.append([mean_squared_error(y_lasso,y_2_train),mean_squared_error(y_lasso_val,y_2_val)])

"""random_forest"""
forest = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_2_train)
y_rf = forest.predict(X_train)
y_rf_embed = forest.predict(X_val)
Loss_res.append([mean_squared_error(y_rf,y_2_train),mean_squared_error(y_rf_embed,y_2_val)])

data= pd.DataFrame([])
data['xg'] = y_xg
data['lasso'] = y_lasso
data['rf'] = y_rf


data_val= pd.DataFrame([])
data_val['xg'] = y_xg_embed
data_val['lasso'] = y_lasso_val
data_val['rf'] = y_rf_embed


X_train_embed = data
data.to_csv("X_embed.csv")
X_val_embed = data_val
print(X_train_embed)


var_num = 3
embed_model = EmbedingModel()
embed_model.build(input_shape=[(None,var_num),(None,20)]) #
embed_model.summary()
optimizer = tf.optimizers.Adam(learning_rate=lr)
embed_model.compile(optimizer, loss='mean_squared_error')
variables = embed_model.trainable_variables
epo = []
los = []
los_val = []
temp_val = np.inf
print('x_train',X_train_embed)


for epoch in range(200):
    with tf.GradientTape() as tape:
        X_train_embed = tf.reshape(X_train_embed,shape=(-1,var_num))
        other20 = tf.reshape(train[charac],shape=(-1,20))
        y_pred = embed_model([X_train_embed,other20]) #call() [1,1] -> [1]  [[1]]->[1]
        y_pred = tf.squeeze(y_pred)
        loss = tf.losses.MSE(y_pred, y_2_train)
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))

    #valid
    X_val_embed = tf.reshape(X_val_embed,shape=(-1,var_num))
    other20_val = tf.reshape(val[charac],shape=(-1,20))
    y_val_pred = embed_model([X_val_embed,other20_val])
    y_val_pred = tf.squeeze(y_val_pred)
    loss_val = tf.losses.MSE(y_val_pred, y_2_val)
    if loss_val<= temp_val:
        temp_val = loss_val
        embed_model.save_weights("min_loss_val.h5")

    #log
    epo.append(epoch)
    los.append(loss.numpy())
    los_val.append(loss_val.numpy())
    print(epoch, "loss:", float(loss),"val_loss",float(loss_val))

Loss_res.append([los[-1],los_val[-1]])
Loss_res = pd.DataFrame(Loss_res).T
Loss_res.columns = ['Xgboost','Lasso','RandomForest','OurMethod']
Loss_res.to_csv('./loss_res.csv')

res = pd.DataFrame([])
res['loss'] = los
res['loss_val'] = los_val
res.to_csv('./Q2/train_loss.csv')

plt.figure(figsize=(8,5))
plt.subplot(111)
plt.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.10)
sns.lineplot(data=res,markers=True)
plt.yticks(rotation = 0,fontsize = 10)
plt.xticks(fontsize=10)
plt.ylabel('MSE',fontsize=12)
plt.xlabel('epoch',fontsize=12)
plt.show()
plt.clf()

"""加载test_dataset"""
molecular = pd.read_excel('Molecular_Descriptor.xlsx',1)
charac_select = ['MDEC-23','BCUTp-1h','ATSp4','ALogP','MLFER_A','minsssN','ATSp1','minHsOH','LipoaffinityIndex','mindO','ATSp5','C1SP2','ATSc4','C3SP2','maxsssCH','maxHBd','minHBint10','ATSc3','maxHsOH','AMR']
X = molecular[charac_select]

y_xg = model_xg.predict(X)
y_lasso = model_lasso.predict(X)
y_rf = forest.predict(X)

data= pd.DataFrame([])
data['xg'] = y_xg
data['lasso'] = y_lasso
data['rf'] = y_rf


data_ = tf.reshape(data,shape=(-1,var_num))
other20_ = tf.reshape(molecular[charac_select],shape=(-1,20))
print(data)
pic50_pred = embed_model([data_,other20_]).numpy()
print(pic50_pred)
pic50= pd.DataFrame([])
pic50['pic50'] = list(pic50_pred)
pic50['xg'] = y_xg
pic50['lasso'] = y_lasso
pic50['rf'] = y_rf

pic50.to_csv('pic50.csv')

# """查看最后一层权重"""
# embed_model = EmbedingModel()
# embed_model.build(input_shape=[(None,3),(None,20)])
# embed_model.load_weights('min_loss_val.h5')
# print(embed_model.fc3.get_weights())

