# -*- coding: utf-8 -*-
# @Time    : 2018/10/3 9:21
# @Author  : pengchenghu
# @FileName: ch10_1.py
# @Software: PyCharm
# @多层感知器进阶： JSON 序列化模型

import time
from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import model_from_json


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

    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
# 构建模型
model = create_model()
model.fit(x, y_labels, epochs=200, batch_size=5, verbose=0)

scores = model.evaluate(x, y_labels, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

# 将模型保存成JSON
model_json = model.to_json()
with open('model.json', 'w') as file:
    file.write(model_json)

# 保存模型权值
model.save_weights('model.json.h5')

print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
# 从JSON文件加载模型
with open('model.json', 'r') as file:
    model_json = file.read()

# 加载模型
new_model = model_from_json(model_json)
new_model.load_weights('model.json.h5')

# 编译模型
new_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# 评估从JSON文件中加载的模型
scores = new_model.evaluate(x, y_labels, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
