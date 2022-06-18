import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import xgboost

def min_max(x):
    print('min:',np.min(x),'max',np.max(x))
    if np.std(x) == 0:  #避免0除
        return x
    return (x-np.mean(x))/np.std(x)

"""read files"""
print('reading files...')
molecular = pd.read_excel('Molecular_Descriptor.xlsx',0)
ER = pd.read_excel('ADMET.xlsx',0)
Q3data = pd.merge(molecular,ER,on='SMILES')
print(Q3data)

mole_columns = molecular.columns
X_all = Q3data[mole_columns[1:-1]]
print(list(X_all.columns))
# 所有自变量归一化处理
STD = True
if STD:
    for name in list(X_all.columns):
        d = min_max(X_all[name])
        X_all = X_all.drop(name, axis=1)
        X_all[name] = d


y_1 = Q3data['Caco-2']
y_2 = Q3data['CYP3A4']
y_3 = Q3data['hERG']
y_4 = Q3data['HOB']
y_5 = Q3data['MN']

name_list = ['Caco-2','CYP3A4','hERG','HOB','MN']

"""变量筛选"""
"""xgboost"""
def scores_plot(score,num,y_name):
    print(score)
    scores = pd.DataFrame([])
    scores['Molecular_Descriptor'] = score.keys()
    scores['Score_'+y_name] = score.values()
    scores = scores.sort_values(by='Score_'+y_name,ascending=False).reset_index()
    scores.to_csv('./Q3/xgboost_Q3_'+y_name + '.csv')
    scores = scores.loc[:num]
    plt.figure(figsize=(8,5))
    plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.27)
    sns.barplot(x='Molecular_Descriptor',y='Score_'+y_name,data=scores)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('Molecular_Descriptor', fontsize=12)
    plt.ylabel('Score_'+y_name, fontsize=12)
    plt.savefig('./Q3/xgboost_im_'+y_name + '.png')
    plt.clf()
#
# model_xg_1 = xgboost.XGBClassifier().fit(X_all, y_1)
# # score = model_xg_1.get_booster().get_score()
# # scores_plot(score,30,'Caco-2')
#
# model_xg_2 = xgboost.XGBRegressor().fit(X_all, y_2)
# # score = model_xg_2.get_booster().get_score()
# # scores_plot(score,30,'CYP3A4')
#
# model_xg_3 = xgboost.XGBRegressor().fit(X_all, y_3)
# # score = model_xg_3.get_booster().get_score()
# # scores_plot(score,30,'hERG')
#
# model_xg_4 = xgboost.XGBRegressor().fit(X_all, y_4)
# # score = model_xg_4.get_booster().get_score()
# # scores_plot(score,30,'HOB')
#
# model_xg_5 = xgboost.XGBRegressor().fit(X_all, y_5)
# # score = model_xg_5.get_booster().get_score()
# scores_plot(score,30,'MN')

"""Lasso"""
def lasso_plot(model,num,y_name):
    coef_lr = pd.DataFrame({'Var' : X_all.columns,
                            'Coef' : model.coef_.flatten()
                            })

    index_sort =  np.abs(coef_lr['Coef']).sort_values(ascending = False).index
    coef_lr_sort = coef_lr.loc[index_sort,:]
    coef_lr_sort = coef_lr_sort.reset_index()

    # 变量重要性柱形图
    plt.figure(figsize=(8,5))
    ax = plt.subplot(111)
    lasso_atri_chose_res = coef_lr_sort[['Var','Coef']]
    lasso_atri_chose_res.to_csv('./Q3/lasso_im_'+y_name+'.csv')
    lasso_atri_chose_res = lasso_atri_chose_res.loc[:num]  # 前30个重要性指标
    print(lasso_atri_chose_res)
    plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.09)
    sns.barplot(x='Coef', y='Var', data=lasso_atri_chose_res, orient='h')
    plt.xticks(rotation = 0,fontsize = 10)
    plt.yticks(fontsize=10)
    plt.xlabel('Coef',fontsize=12)
    plt.ylabel('Molecular_Descriptor',fontsize=12)
    plt.show()
    # plt.savefig('./Q3/lasso_im_'+y_name+'.png')
    plt.clf()

from sklearn.linear_model import Lasso
lr_1 =  Lasso(alpha = 0.1)
lr_1.fit(X_all,y_1)
# lasso_plot(lr_1,30,'Caco-2')

lr_2 =  Lasso(alpha = 0.1)
lr_2.fit(X_all,y_2)
lasso_plot(lr_2,30,'CYP3A4')

lr_3 =  Lasso(alpha = 0.1)
lr_3.fit(X_all,y_3)
# lasso_plot(lr_3,30,'hERG')

lr_4 =  Lasso(alpha = 0.1)
lr_4.fit(X_all,y_4)
# lasso_plot(lr_4,30,'HOB')

lr_5 =  Lasso(alpha = 0.1)
lr_5.fit(X_all,y_5)
# lasso_plot(lr_5,30,'MN')

"""随机森林"""
from sklearn.ensemble import RandomForestRegressor
def rf_plot(importances,num,y_name):
    indices = np.argsort(importances)[::-1]
    res = {}
    for f in range(X_all.shape[1]):
        res[X_all.columns[indices[f]]] = importances[indices[f]]
    ress = pd.DataFrame([])
    ress['Molecular_Descriptor'] = res.keys()
    ress['importance'] = res.values()
    ress.to_csv('./Q3/rf_im_'+y_name+'.csv')
    ress = ress.loc[:num]   #输出前30个
    plt.figure(figsize=(8,5))
    plt.subplot(111)
    plt.subplots_adjust(left=0.075, right=0.98, top=0.98, bottom=0.27)
    sns.barplot(x='Molecular_Descriptor', y='importance', data=ress)
    plt.xticks(rotation = 90,fontsize = 10)
    plt.yticks(fontsize=10)
    plt.xlabel('Molecular_Descriptor',fontsize=12)
    plt.ylabel('importance',fontsize=12)
    plt.savefig('./Q3/rf_im_'+y_name+'.png')
    plt.clf()

# forest_1 = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1).fit(X_all, y_1)
# # importances = forest_1.feature_importances_
# # rf_plot(importances,30,'Caco-2')
#
# forest_2 = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1).fit(X_all, y_2)
# # importances = forest_2.feature_importances_
# # rf_plot(importances,30,'CYP3A4')
#
# forest_3 = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1).fit(X_all, y_3)
# # importances = forest_3.feature_importances_
# # rf_plot(importances,30,'hERG')
#
# forest_4 = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1).fit(X_all, y_4)
# # importances = forest_4.feature_importances_
# # rf_plot(importances,30,'HOB')
#
# forest_5 = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1).fit(X_all, y_5)
# # importances = forest_5.feature_importances_
# # rf_plot(importances,30,'MN')

"""基于MIV的神经网络变量筛选"""
def miv_plot(model_,num,y_name,train,val,model_return):
    y_train = train[y_name]
    X_train = train[mole_columns[1:-1]]
    y_val = val[y_name]
    X_val = val[mole_columns[1:-1]]

    y_train = tf.one_hot(y_train,depth=2) #sparse_catergory_crossentropy
    y_val = tf.one_hot(y_val,depth=2)
    model_.fit(x=X_train, y=y_train, epochs=100,batch_size=64,
          verbose=1,
          validation_data=(X_val, y_val))
    if model_return == False:
        score_train = model_.evaluate(X_train, y_train, verbose=0)
        score_val = model_.evaluate(X_val, y_val, verbose=0)
        print('training loss:',score_train[0],'train acc:',score_train[1])
        print('valid loss:',score_val[0]," valid acc:",score_val[1])
        minus = {}
        total_num = len(mole_columns[1:-1])
        step = 1
        for name_ in mole_columns[1:-1]:
            d = X_train[name_]*1.1
            X_train = X_train.drop(name_,axis =1)
            X_train[name_] = d
            y_increase = model_.predict(X_train)

            e = X_train[name_] * 0.9/1.1
            X_train = X_train.drop(name_,axis =1)
            X_train[name_] = e
            y_decrease = model_.predict(X_train)

            minus[name_] = np.mean(y_increase-y_decrease)
            print(np.mean(y_increase-y_decrease),'{0}/{1}'.format(step,total_num))
            step+=1
        minuss = pd.DataFrame([])
        minuss['Molecular_Descriptor'] = minus.keys()
        minuss['MIV_'+y_name] = minus.values()
        minuss.to_csv('./Q3/miv_im_'+y_name+'.csv')
        minuss = minuss.iloc[minuss['MIV_'+y_name].abs().argsort()][::-1]
        minuss = minuss.reset_index()
        minuss = minuss.loc[:num]
        print(minuss)
        plt.figure(figsize=(8,5))
        plt.subplot(111)
        plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.1)
        sns.barplot(y='Molecular_Descriptor', x='MIV_'+y_name, data=minuss,orient='h' )
        plt.yticks(rotation = 0,fontsize = 10)
        plt.xticks(fontsize=10)
        plt.ylabel('Molecular_Descriptor',fontsize=12)
        plt.xlabel('MIV_'+y_name,fontsize=12)
        plt.savefig('./Q3/miv_im_'+y_name+'.png')
        plt.clf()
    else:
        return model_

import tensorflow as tf
from tensorflow.keras import layers,Model,Sequential
from sklearn.model_selection import train_test_split

model = Sequential([
    layers.Dense(128,activation='relu',use_bias=False),
    layers.Dense(32,activation='relu',use_bias=False),
    layers.Dense(2)
])

model.build(input_shape=(None,728))
model.compile( loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              metrics=['accuracy'])
model.summary()
train,val = train_test_split(Q3data, test_size=0.2)

name_list = ['Caco-2','CYP3A4','hERG','HOB','MN']

for name in name_list:
    miv_plot(model,30,name,train,val,model_return=False)


