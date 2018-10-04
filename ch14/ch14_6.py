# -*- coding: utf-8 -*-
# @Time    : 2018/10/4 21:13
# @Author  : pengchenghu
# @FileName: ch14_6.py
# @Software: PyCharm
# @Keras 中的图像增强

from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras import backend
import os

backend.set_image_data_format('channels_first')

# 从Keras 导入Mnist 数据
(x_train, y_train), (x_validation, y_validation) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')  # 个数、通道数、宽度、高度
x_validation = x_validation.reshape(x_validation.shape[0], 1, 28, 28).astype('float32')

# ZAC白化
imgGen = ImageDataGenerator(zca_whitening=True)
imgGen.fit(x_train)

# 创建目录并保存
try:
    os.mkdir('image/zca/')
except:
    print('The fold is exist')
    for x_batch, y_batch in imgGen.flow(x_train, y_train, batch_size=9, save_to_dir='image/zca/', save_prefix='zca',\
                                        save_format='png'):
        break

# 特征值标准化
imgGen1 = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# 图像旋转
imgGen2 = ImageDataGenerator(rotation_range=90)
# 图像移动
imgGen3 = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2)
# 图像剪切
imgGen4 = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
# 图像反转
imgGen5 = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
