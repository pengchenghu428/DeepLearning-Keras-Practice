# -*- coding: utf-8 -*-
# @Time    : 2018/9/24 11:35
# @Author  : pengchenghu
# @FileName: ch06_1.py
# @Software: PyCharm
# @使用价差验证评估模型

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 构建模型
def create_model():
    # 构建模型
    model = Sequential()
    model.add(Dense(units=12, input_dim=8, activation='relu'))  # 第一层隐藏层有12个神经元，在此之前的表示层有8个神经元
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    # 编译模型
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

seed = 7
# 设定随机数种子
np.random.seed(seed)

# 导入数据
dataset = np.loadtxt('datasets/pima-indians-diabetes-data.csv', delimiter=',')
# 分割输入变量x和输出变量y
x = dataset[:, 0:8]
y = dataset[:, 8]

# 创建模型
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)

# 10折交叉验证
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, x, y, cv=kfold)
print(results.mean())

