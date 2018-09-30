# -*- coding: utf-8 -*-
# @Time    : 2018/9/30 10:38
# @Author  : pengchenghu
# @FileName: ch09.py
# @Software: PyCharm
# @二分类实例：银行营销分类
import time
import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 导入数据并转化成数字
dataset = pd.read_csv('bank/bank.csv', delimiter=';')
dataset['job'] = dataset['job'].replace(to_replace=['admin.', 'unknown', 'unemployed', 'management',
                                                    'housemaid', 'entrepreneur', 'student', 'blue-collar',
                                                    'self-employed', 'retired', 'technician', 'services'],
                                        value=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
dataset['marital'] = dataset['marital'].replace(to_replace=['married', 'single', 'divorced'], value=[0, 1, 2])
dataset['education'] = dataset['education'].replace(to_replace=['unknown', 'secondary', 'primary', 'tertiary'],
                                                    value=[0, 2, 1, 3])
dataset['default'] = dataset['default'].replace(to_replace=['no', 'yes'], value=[0, 1])
dataset['housing'] = dataset['housing'].replace(to_replace=['no', 'yes'], value=[0, 1])
dataset['loan'] = dataset['loan'].replace(to_replace=['no', 'yes'], value=[0, 1])
dataset['contact'] = dataset['contact'].replace(to_replace=['cellular', 'unknown', 'telephone'], value=[0, 1, 2])
dataset['poutcome'] = dataset['poutcome'].replace(to_replace=['unknown', 'other', 'success', 'failure'],
                                                  value=[0, 1, 2, 3])
dataset['month'] = dataset['month'].replace(to_replace=['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                                        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                                            value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
dataset['y'] = dataset['y'].replace(to_replace=['no', 'yes'], value=[0, 1])

# 分离输入与输出
array = dataset.values
x = array[:, 0:16]
y = array[:, 16]

# 设置随机数种子
seed = 7
np.random.seed(seed)


# 构建模型函数
def create_model(units_list=[16], optimizer='adam', init='normal'):
    # 构建模型
    model = Sequential()

    # 构建第一个隐藏层和输入层
    units = units_list[0]
    model.add(Dense(units=units, activation='relu', input_dim=16, kernel_initializer=init))

    # 构建更多隐藏层
    for units in units_list[1:]:
        model.add(Dense(units=units, activation='relu', kernel_initializer=init))

    model.add(Dense(units=1, activation='sigmoid', kernel_initializer=init))

    # 编译模型
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
# 用于Scikit-Leran分类模型Keras包装类KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=200, batch_size=5, verbose=0)

# 采用十折交叉验证来评估简单的神经网络模型
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, x, y, cv=kfold)
print('Accuracy: %.2f%% (%.2f)' % (results.mean() * 100, results.std()))

print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

new_x = StandardScaler().fit_transform(x)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, new_x, y, cv=kfold)
print('Accuracy: %.2f%% (%.2f)' % (results.mean() * 100, results.std()))

print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
# 调参选择最优模型
param_grid = {}
param_grid['units_list'] = [[16], [30], [16, 8], [30, 8]]
# 调参
grid = GridSearchCV(estimator=model, param_grid=param_grid)
results = grid.fit(new_x, y)

# 输出结果
print('Best: %f using %s' % (results.best_score_, results.best_params_))
means = results.cv_results_['mean_test_score']
stds = results.cv_results_['std_test_score']
params = results.cv_results_['params']

for mean, std, param in zip(means, stds, params):
    print('%f (%f) with: %r' % (mean, std, param))

print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))