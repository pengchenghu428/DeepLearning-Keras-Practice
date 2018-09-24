# -*- coding: utf-8 -*-
# @Time    : 2018/9/24 22:01
# @Author  : pengchenghu
# @FileName: ch07.py
# @Software: PyCharm
# @多酚类实例：鸢尾花分类

from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import  cross_val_score
from sklearn.model_selection import KFold
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 导入数据
dataset = datasets.load_iris()

x = dataset.data
y = dataset.target

# 设定随机数种子
seed = 8
np.random.seed(seed)


# 构建模型
def create_model(optimizer='adam', init='glorot_uniform'):
    # 构建模型
    model = Sequential()
    model.add(Dense(units=4, kernel_initializer=init, input_dim=4, activation='relu'))
    model.add(Dense(units=6, kernel_initializer=init, activation='relu'))
    model.add(Dense(units=3, kernel_initializer=init, activation='softmax'))

    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


model = KerasClassifier(build_fn=create_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, x, y, cv=kfold)
print('Accuracy: %.2f%% (%.2f)' % (results.mean()*100, results.std()))
