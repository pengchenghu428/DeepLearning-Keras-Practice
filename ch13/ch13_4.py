# -*- coding: utf-8 -*-
# @Time    : 2018/10/4 19:46
# @Author  : pengchenghu
# @FileName: ch13_4.py
# @Software: PyCharm
# @手写数字识别：简单卷积神经网络

from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend

# 设定随机数种子
seed = 7
np.random.seed(seed)

# 从Keras 导入Mnist 数据
(x_train, y_train), (x_validation, y_validation) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')  # 个数、通道数、宽度、高度
x_validation = x_validation.reshape(x_validation.shape[0], 28, 28, 1).astype('float32')

# 格式化数据到0~1
x_train = x_train / 255
x_validation = x_validation / 255

# 进行one-hot编码
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)


# 创建模型
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))

    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = create_model()
filepath = 'ch13_4-weights.best.h5'
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callback_list = [checkpoint]
model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=20, batch_size=200, verbose=2, callbacks=callback_list)

score  = model.evaluate(x_validation, y_validation)
print('CNN_Small: %.2f%%' % (score[1] * 100))
