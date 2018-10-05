# -*- coding: utf-8 -*-
# @Time    : 2018/10/5 20:17
# @Author  : pengchenghu
# @FileName: 16_2.py
# @Software: PyCharm
# @情感分析实例：IMDB 影评情感分析 - 导入数据

import os
from keras.datasets import imdb
import numpy as np
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':    # 针对不同环境，使用不同的diaplay
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from matplotlib import pyplot as plt

# Keras 提供的数据集将单词转化成整数，这个整数代表单词在整个数据集中的流行程度
(x_train, y_train), (x_validation, y_validation) = imdb.load_data()

# 合并训练数据集和评估集
x = np.concatenate((x_train, x_validation), axis=0)
y = np.concatenate((y_train, y_validation), axis=0)

print('x shape is %s, y shape is %s' % (x.shape, y.shape))
print('Classes: %s' % len(np.unique(y)))

print('Total words: %s' % len(np.unique(np.hstack(x))))

result = [len(word) for word in x]
print('Mean: %.2f words (STD: %.2f)' % (np.mean(result), np.std(result)))

# 图标保存
plt.subplot(121)
plt.boxplot(result)
plt.subplot(122)
plt.hist(result)  # 绘制直方图
plt.savefig('imdb_display.png')
plt.close()
