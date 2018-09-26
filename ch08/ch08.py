# -*- coding: utf-8 -*-
# @Time    : 2018/9/26 22:56
# @Author  : pengchenghu
# @FileName: ch08.py
# @Software: PyCharm
# @回归问题实例：波士顿房价预测

import os
import time
from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 导入数据
dataset = datasets.load_boston()

x = dataset.data
y = dataset.target

# 设定随机数种子
seed = 7
np.random.seed(seed)


# 构建模型函数
def create_model(units_list=[13], optimizer= 'adam', init='normal'):
    # 构建模型
    model = Sequential()

    # 构建第一个隐藏层和输入层
    units = units_list[0]
    model.add(Dense(units=units, activation='relu', input_dim=13, kernel_initializer=init))
    # 构建更多的隐藏层
    for units in units_list[1:]:
        model.add(Dense(units=units, activation='relu', kernel_initializer=init))

    model.add(Dense(units=1, kernel_initializer=init))

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

model = KerasRegressor(build_fn=create_model, epochs=200, batch_size=5, verbose=0)

# 数据标准化，改进算法
steps = []
steps.append(('standardize', StandardScaler()))
steps.append(('mlp', model))
pipeline = Pipeline(steps)
# 设置算法评估基准
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, x, y, cv=kfold)
print('Baseline: %.2f (%.2f) MSE' % (results.mean(), results.std()))

print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
