# -*- coding: utf-8 -*-
# @Time    : 2018/10/3 10:27
# @Author  : pengchenghu
# @FileName: ch10_3.py
# @Software: PyCharm
# @多层感知器进阶：模型增量更新

import time
from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import model_from_json
from sklearn.model_selection import train_test_split


# 设定随机数种子
seed = 7
np.random.seed(seed)

# 导入数据
dataset = datasets.load_iris()

x = dataset.data
y = dataset.target

x_train,x_increment, y_train, y_increment = train_test_split(x, y, test_size=0.2, random_state=seed)

# 将标签转换成分类编码
y_train_labels = to_categorical(y_train, num_classes=3)


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
model.fit(x_train, y_train_labels, epochs=10, batch_size=5, verbose=0)

scores = model.evaluate(x_train, y_train_labels, verbose=0)
print('Base %s: %.2f%%' % (model.metrics_names[1], scores[1]*100))

# 将模型保存成JSON
model_json = model.to_json()
with open('model_3.json', 'w') as file:
    file.write(model_json)

# 保存模型权值
model.save_weights('model_3.json.h5')

print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
# 从JSON文件加载模型
with open('model_3.json', 'r') as file:
    model_json = file.read()

# 加载模型
new_model = model_from_json(model_json)
new_model.load_weights('model_3.json.h5')

# 编译模型
new_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# 增量训练模型
y_increment_labels = to_categorical(y_increment, num_classes=3)
new_model.fit(x_increment, y_increment_labels, epochs=10, batch_size=5, verbose=2)
scores = new_model.evaluate(x_increment, y_increment_labels, verbose=0)
print('Increment %s: %0.2f%%' % (model.metrics_names[1], scores[1]*100))


