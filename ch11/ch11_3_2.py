# -*- coding: utf-8 -*-
# @Time    : 2018/10/3 16:40
# @Author  : pengchenghu
# @FileName: ch11_3_2.py
# @Software: PyCharm
# @Dropout与学习衰减率：学习率线性衰减

import time
from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import LearningRateScheduler
from math import pow, floor


# 导入数据
dataset = datasets.load_iris()

x = dataset.data
y = dataset.target

# 设定随机数种子
seed = 7
np.random.seed(seed)


# 计算学习率
def step_decay(epoch):
    init_lrate = 0.1
    drop = 0.5
    epochs_drop = 10
    lrate = init_lrate*pow(drop, floor((1+epoch)/epochs_drop))
    print('model epoch: %d learning rate: %.8f' % (epoch, lrate))
    return lrate


# 构建模型函数
def create_model(init='glorot_uniform'):
    # 构建模型
    model = Sequential()
    model.add(Dense(units=4, activation='relu', kernel_initializer=init))
    model.add(Dense(units=6, activation='relu', kernel_initializer=init))
    model.add(Dense(units=3, activation='softmax', kernel_initializer=init))

    # 模型优化
    learningRate = 0.1
    momentum = 0.9
    decay_rate = 0.00
    sgd = SGD(lr=learningRate, momentum=momentum, decay=decay_rate, nesterov=False)

    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
lrate = LearningRateScheduler(step_decay)
model = KerasClassifier(build_fn=create_model, epochs=200, batch_size=5, verbose=1, callbacks=[lrate])
model.fit(x, y)
