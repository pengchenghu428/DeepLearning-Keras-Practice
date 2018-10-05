# -*- coding: utf-8 -*-
# @Time    : 2018/10/5 18:23
# @Author  : pengchenghu
# @FileName: ch15_5.py
# @Software: PyCharm
# @图像识别实例：CIFAR-10分类(改进模型)

import keras
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.initializers import RandomNormal
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard

batch_size = 128
epochs = 200
iterations = 391
num_classes = 10
dropout = 0.5
log_filepath = './nin'


def normalize_preprocessing(x_train, x_validation):
    x_train = x_train.astype('float32')
    x_validation = x_validation.astype('float32')

    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:, :, :, 1] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_validation[:, :, :, 1] = (x_validation[:, :, :, i] - mean[i]) / std[i]

    return x_train, x_validation


def scheduler(epoch):
    if epoch <= 60:
        return 0.05
    if epoch <=120:
        return 0.01
    if epoch <= 160:
        return 0.002
    return 0.0004


def build_model():
    model = Sequential()
    model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001), \
                     kernel_initializer=RandomNormal(stddev=0.01), input_shape=x_train.shape[1:], activation='relu'))
    model.add(Conv2D(160, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001), \
                     kernel_initializer=RandomNormal(stddev=0.05), activation='relu'))
    model.add(Conv2D(96, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001), \
                     kernel_initializer=RandomNormal(stddev=0.05), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Dropout(rate=dropout))
    model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001), \
                     kernel_initializer=RandomNormal(stddev=0.05), activation='relu'))
    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001), \
                     kernel_initializer=RandomNormal(stddev=0.05), activation='relu'))
    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001), \
                     kernel_initializer=RandomNormal(stddev=0.05), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Dropout(rate=dropout))
    # p145对模型的网络拓扑结构描述是5*5的感受野，但p148代码使用的是3*3的感视野
    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001), \
                     kernel_initializer=RandomNormal(stddev=0.05), activation='relu'))
    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001), \
                     kernel_initializer=RandomNormal(stddev=0.05), activation='relu'))
    model.add(Conv2D(10, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001), \
                     kernel_initializer=RandomNormal(stddev=0.05), activation='relu'))
    '''
    global average pooling 主要用来解决全连接问题，其主要是将最后一层的特征图进行整张图的一个均值池化，
    形成一个特征点，将这些特征点组成最后的特征向量进行softmax中计算
    '''
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))   # 等价于 model.add(Dense(units=num_classes, activation='softmax'))

    sgd = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    np.random.seed(seed=7)
    # 导入数据
    (x_train, y_train), (x_validation, y_validation) = cifar10.load_data()
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_validation = keras.utils.np_utils.to_categorical(y_validation, num_classes)

    x_train, x_validation = normalize_preprocessing(x_train, x_validation)

    # 构建神经网络
    model = build_model()
    print(model.summary())  # 打印模型概况

    # 设置回调函数，实现学习率衰减
    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr, tb_cb]

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=cbks, \
              validation_data=(x_validation, y_validation), verbose=2)
    # 同时保存model和weight的方式
    model.save('nin.h5')