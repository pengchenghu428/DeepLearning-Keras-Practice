#coding=utf-8

from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np

# 设定随机数种子
np.random.seed(7)

# 导入数据
dataset = np.loadtxt('datasets/pima-indians-diabetes-data.csv', delimiter=',')
# 分割输入变量x和输出变量y
x = dataset[:, 0:8]
y = dataset[:, 8]

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
# epochs: 数据里固定次数的迭代
# batch_size: 权重更新的每个批次中所用的实例个数
# validation_split: 验证分割参数设置为0.2
model.fit(x=x, y=y, epochs=150, batch_size=10, validation_split=0.2)

# 评估模型
scores = model.evaluate(x=x, y=y)
print('\n%s : %.2f%%' % (model.metrics_names[1], scores[1]*100))

