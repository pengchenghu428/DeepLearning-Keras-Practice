# -*- coding: utf-8 -*-
# @Time    : 2018/10/3 11:09
# @Author  : pengchenghu
# @FileName: ch10_5.py
# @Software: PyCharm
# @多层感知器进阶：模型训练过程可视化

import os
import time
from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':    # 针对不同环境，使用不同的diaplay
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from matplotlib import pyplot as plt


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
history = model.fit(x, y_labels, validation_split=0.2, epochs=200, batch_size=5, verbose=0)

# 评估模型
scores = model.evaluate(x, y_labels, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

# history列表
print(history.history.keys())

# accuracy的历史
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.savefig('figures/accuracy.png')
plt.close()

# loss的历史
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.savefig('figures/loss.png')
