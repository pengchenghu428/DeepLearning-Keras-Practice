# -*- coding: utf-8 -*-
# @Time    : 2018/9/21 10:50
# @Author  : pengchenghu
# @FileName: ch05_k_evaluation.py
# @Software: PyCharm

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed = 7
# 设定随机数种子
np.random.seed(seed)

# 导入数据
dataset = np.loadtxt('datasets/pima-indians-diabetes-data.csv', delimiter=',')
# 分割输入变量x和输出变量y
x = dataset[:, 0:8]
y = dataset[:, 8]

# K折交叉验证集
# n_split: 子集个数
# random_state: 随机种子
# shuffle: 是否打乱子集顺序
kfold = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
cvscores = []

for train, validation in kfold.split(x, y):
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
    # epochs: 数据里固定次数的迭代
    # batch_size: 权重更新的每个批次中所用的实例个数
    # verbose=0：关闭fit()和evaluate()函数的详细输出
    model.fit(x[train], y[train], epochs=150, batch_size=10, verbose=0)

    # 评估模型
    scores = model.evaluate(x=x[validation], y=y[validation], verbose=0)
    cvscores.append(scores[1]*100)

    # 输出评估结果
    print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))

print('%.2f%% (+/- %.2f%%)' % (np.mean(cvscores), np.std(cvscores)))
