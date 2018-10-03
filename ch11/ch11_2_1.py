# -*- coding: utf-8 -*-
# @Time    : 2018/10/3 15:17
# @Author  : pengchenghu
# @FileName: ch11_2_1.py
# @Software: PyCharm
# @Dropout 与学习率衰减：在输入层使用Dropout

import time
from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# 导入数据
dataset = datasets.load_iris()

x = dataset.data
y = dataset.target

# 设定随机数种子
seed = 7
np.random.seed(seed)


# 构建模型函数
def create_model(init='glorot_uniform'):
    # 构建模型
    model = Sequential()
    model.add(Dropout(rate=0.2, input_shape=(4,)))  # 添加rate=0.2, input_shape=(4, )的输入层
    model.add(Dense(units=4, activation='relu', kernel_initializer=init))
    model.add(Dense(units=6, activation='relu', kernel_initializer=init))
    model.add(Dense(units=3, activation='softmax', kernel_initializer=init))

    # 定义Dropout
    # 随机梯度下降算法
    # lr：大于0的浮点数，学习率
    # momentum：大于0的浮点数，动量参数
    # decay：大于0的浮点数，每次更新后的学习率衰减值
    # nesterov：布尔值，确定是否使用Nesterov动量
    sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)

    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
model = KerasClassifier(build_fn=create_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, x, y, cv=kfold)
print('Accuracy: %.2f%% (%.2f)' % (results.mean()*100, results.std()))
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
