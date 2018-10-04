# -*- coding: utf-8 -*-
# @Time    : 2018/10/4 19:20
# @Author  : pengchenghu
# @FileName: ch13_3.py
# @Software: PyCharm
# @手写数字识别：多层感知器模型
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

# 从Keras 导入Mnist 数据
(x_train, y_train), (x_validation, y_validation) = mnist.load_data()

# 设定随机数种子
seed = 7
np.random.seed(seed)

num_pixels = x_train.shape[1] * x_train.shape[2]  # 每张图片的像素
print(num_pixels)
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')  # 将图片转换成一维向量
x_validation = x_validation.reshape(x_validation.shape[0], num_pixels).astype('float32')

# 格式化数据到0~1
x_train = x_train / 255
x_validation = x_validation / 255

# 进行one-hot编码
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)
num_classes = y_train.shape[1]
print(num_classes)


# 定义基准MLP模型
def create_model():
    # 创建模型
    model = Sequential()
    model.add(Dense(units=num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=num_classes, kernel_initializer='normal', activation='softmax'))

    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = create_model()
filepath = 'ch13_3-weights.best.h5'
checkpoint=ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callback_list = [checkpoint]
model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=20, batch_size=200, callbacks=callback_list)

score  = model.evaluate(x_validation, y_validation)
print('MLP: %.2f%%' % (score[1] * 100))
