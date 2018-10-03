# -*- coding: utf-8 -*-
# @Time    : 2018/10/3 10:47
# @Author  : pengchenghu
# @FileName: ch10_4.py
# @Software: PyCharm
# @多层感知器进阶：神经网络的检查点

import time
from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint


# 导入数据
dataset = datasets.load_iris()

x = dataset.data
y = dataset.target

# 将标签转换成分类编码
y_labels = to_categorical(y, num_classes=3)

# 设定随机数种子
seed = 7
np.random.seed(seed)


# 构建模型函数
def create_model(optimizer='rmsprop', init='glorot_uniform'):
    # 构建模型
    model = Sequential()
    model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
    model.add(Dense(units=6, activation='relu', kernel_initializer=init))
    model.add(Dense(units=3, activation='softmax', kernel_initializer=init))

    # 加载权重
    filepath = 'checkpoints/weights.best.h5'
    model.load_weights(filepath=filepath)

    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
# 构建模型
model = create_model()

# 评估模型
scores = model.evaluate(x, y_labels, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
