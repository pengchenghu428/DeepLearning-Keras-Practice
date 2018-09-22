# -*- coding: utf-8 -*-
# @Time    : 2018/9/21 10:20
# @Author  : pengchenghu
# @FileName: ch05_manual_evaluation.py
# @Software: PyCharm

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

seed = 7
# 设定随机数种子
np.random.seed(seed)

# 导入数据
dataset = np.loadtxt('datasets/pima-indians-diabetes-data.csv', delimiter=',')
# 分割输入变量x和输出变量y
x = dataset[:, 0:8]
y = dataset[:, 8]

# 分割输入变量x和输出变量y
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2, random_state=seed)

# 创建模型
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))  # 第一层包含12个神经元，输入维度8，激活函数RELU
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # 二分类问题采用sigmoid作为激活函数效果好

# 编译模型
# 在编译模型时，必须指定用于评估的一组权重的损失函数（loss）、用于搜索网络的不同权重的优化器（optimizer），以及希望在模型训练期间搜集的 \
# 和报告的可选指标
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # 二进制交叉熵、Adam梯度下降算法、精确度作为度量模型的标准

# 训练模型并自动评估模型
# x_train, y_train: 训练集
# x_validation, y_validation: 验证集
# epochs: 数据里固定次数的迭代
# batch_size: 权重更新的每个批次中所用的实例个数
model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=150, batch_size=10)

# 评估模型
scores = model.evaluate(x=x, y=y)
print('\n%s : %.2f%%' % (model.metrics_names[1], scores[1]*100))

