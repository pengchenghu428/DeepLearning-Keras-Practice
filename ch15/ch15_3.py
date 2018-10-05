# -*- coding: utf-8 -*-
# @Time    : 2018/10/5 17:42
# @Author  : pengchenghu
# @FileName: ch15_3.py
# @Software: PyCharm
# @图像识别实例：CIFAR-10分类(简单卷积神经网络)

from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from scipy.misc import toimage
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras import backend

backend.set_image_data_format('channels_first')

# 导入数据
(x_train, y_train), (x_validation, y_validation) = cifar10.load_data()

# 设定随机数种子
seed = 7
np.random.seed(seed)

# 格式化数据0~1
x_train = x_train.astype('float32')
x_validation = x_validation.astype('float32')
x_train = x_train / 255
x_validation = x_validation / 255

# 进行one-hot 编码
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)
num_classes = y_train.shape[1]


# 构建基准模型
def create_model(epochs=25):
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=num_classes, activation='softmax'))

    lrate = 0.01
    decay = lrate / epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


epochs = 25
model = create_model(epochs=epochs)
filepath = 'ch15_3-weights.best.h5'
checkpoint = ModelCheckpoint(filepath=filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
callback_list = [checkpoint]
model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=32, verbose=2, callbacks=callback_list)
scores = model.evaluate(x=x_validation, y=y_validation, verbose=0)
print('CNN_Small Accuracy: %.2f%%' % (scores[1] * 100))
