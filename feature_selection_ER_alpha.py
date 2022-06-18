import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#字体设置
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
import xgboost
import seaborn as sns

def min_max(x):
    print('min:',np.min(x),'max',np.max(x))
    if np.std(x) == 0:  #避免0除
        return x
    return (x-np.mean(x))/np.std(x)

def scores_plot(score,y_name):
    print(score)
    scores = pd.DataFrame([])
    scores['Molecular_Descriptor'] = score.keys()
    scores['Score_'+y_name] = score.values()
    scores = scores.sort_values(by='Score_'+y_name,ascending=False).reset_index()
    scores.to_csv('xgboost.csv')
    plt.figure(figsize=(10,7))
    plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.18)
    sns.barplot(x='Molecular_Descriptor',y='Score_'+y_name,data=scores)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('Molecular_Descriptor', fontsize=12)
    plt.ylabel('Score_'+y_name, fontsize=12)
    plt.show()
    plt.clf()

"""read files"""
molecular = pd.read_excel('Molecular_Descriptor.xlsx',0)
ER = pd.read_excel('ERα_activity.xlsx',0)
Q1data = pd.merge(molecular,ER,on='SMILES')
print(Q1data)
mole_columns = molecular.columns
X = Q1data[mole_columns[1:-1]]
print(list(X.columns))

#所有自变量归一化处理
STD = True
if STD :
    for name in list(X.columns) :
        d = min_max(X[name])
        X = X.drop(name, axis=1)
        X[name] = d
print(X)
y_2 = Q1data['pIC50']


"""Xgboost"""
model = xgboost.XGBRegressor().fit(X, y_2)
score = model.get_booster().get_score()
scores_plot(score,'pIC50')


"""lasso"""
from sklearn.linear_model import Lasso
lr2 =  Lasso(alpha = 0.1)
lr2.fit(X,y_2)
lr2.predict(X)

coef_lr2 = pd.DataFrame({'Var' : X.columns,
                        'Coef' : lr2.coef_.flatten()
                        })
index_sort2 =  np.abs(coef_lr2['Coef']).sort_values(ascending = False).index
coef_lr_sort2 = coef_lr2.loc[index_sort2,:]
coef_lr_sort2 = coef_lr_sort2.reset_index().loc[:30]# 前30个重要性指标

# 变量重要性柱形图
plt.figure(figsize=(10,7))
plt.subplot(111)
lasso_atri_chose_res2 = coef_lr_sort2[['Var','Coef']]
# lasso_atri_chose_res2.to_csv("lasso.csv")
print(lasso_atri_chose_res2)
plt.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.07)
sns.barplot(x='Coef', y='Var', data=lasso_atri_chose_res2, orient='h')
plt.xticks(rotation = 0,fontsize = 10)
plt.yticks(fontsize=10)
plt.xlabel('Coef',fontsize=12)
plt.ylabel('Molecular_Descriptor',fontsize=12)
plt.show()
plt.clf()

"""随机森林"""
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
forest.fit(X, y_2)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
res = {}
for f in range(X.shape[1]):
    res[X.columns[indices[f]]] = importances[indices[f]]
ress = pd.DataFrame([])
ress['Molecular_Descriptor'] = res.keys()
ress['importance'] = res.values()
ress.to_csv('./rf.csv')
ress = ress.loc[:30]   #输出前30个
plt.figure(figsize=(10,7))
plt.subplot(111)
plt.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.07)
sns.barplot(x='Molecular_Descriptor', y='importance', data=ress)
plt.xticks(rotation = 90,fontsize = 10)
plt.yticks(fontsize=10)
plt.xlabel('Molecular_Descriptor',fontsize=12)
plt.ylabel('importance',fontsize=12)
plt.show()
plt.clf()

"""基于MIV的神经网络变量筛选"""
def miv_plot(model,num,y_name):
    minus = {}
    total_num = len(mole_columns[1:-1])
    step = 1
    for name in mole_columns[1:-1]:
        d = X_train[name]*1.1
        X_train = X_train.drop(name,axis =1)
        X_train[name] = d
        y_increase = model.predict(X_train)

        e = X_train[name] * 0.9/1.1
        X_train = X_train.drop(name,axis =1)
        X_train[name] = e
        y_decrease = model.predict(X_train)

        minus[name] = np.mean(y_increase-y_decrease)
        print(np.mean(y_increase-y_decrease),'{0}/{1}'.format(step,total_num))
        step+=1
    minuss = pd.DataFrame([])
    minuss['Molecular_Descriptor'] = minus.keys()
    minuss['MIV_pIC50'] = minus.values()
    minuss.to_csv('./miv_im_'+y_name+'.csv')
    minuss = minuss.iloc[minuss['MIV_pIC50'].abs().argsort()][::-1]
    minuss = minuss.reset_index()
    minuss = minuss.loc[:num]
    print(minuss)
    plt.figure(figsize=(8,5))
    plt.subplot(111)
    plt.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.07)
    sns.barplot(y='Molecular_Descriptor', x='MIV_pIC50', data=minuss,orient='h' )
    plt.yticks(rotation = 0,fontsize = 10)
    plt.xticks(fontsize=10)
    plt.ylabel('Molecular_Descriptor',fontsize=12)
    plt.xlabel('MIV_pIC50',fontsize=12)
    plt.show()
    plt.clf()

import tensorflow as tf
from tensorflow.keras import layers,Model,Sequential
from sklearn.model_selection import train_test_split

model = Sequential([
    layers.Dense(50,activation='sigmoid',use_bias=False),
    layers.Dense(1)
])
model.build(input_shape=(1,728))
optimizer = tf.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer, loss='mean_squared_error')
model.summary()
train,val = train_test_split(Q1data, test_size=0.001)
y_2_train = train['pIC50']
X_train = train[mole_columns[1:-1]]
y_2_val = val['pIC50']
X_val = val[mole_columns[1:-1]]
print(X_train)
print("training....")
model.fit(x=X_train,y=y_2_train,epochs=5)
miv_plot(model,30,'MIV_pIC50')




