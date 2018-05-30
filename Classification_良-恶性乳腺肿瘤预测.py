# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     13-良-恶性乳腺肿瘤数据预处理
   Author :       Zeke
   date：          2018/5/30
   Description :   多元线性回归
-------------------------------------------------
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

cols = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size'
        'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
        'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', names=cols)
# print(data.head())

# 将？替换为缺失值
data = data.replace('?', np.nan)
# 丢弃带缺失的数据
data = data.dropna(how='any')

print(data.shape)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:9],
                                                    data.iloc[:,9],
                                                    test_size=0.3,
                                                    random_state=123)
print(y_train.value_counts())
print(y_test.value_counts())

# 标准化数据
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 数据拟合
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_y_predict = lr.predict(X_test)

#性能分析
from sklearn.metrics import classification_report
print(classification_report(y_test, lr_y_predict, target_names=['Benign','Malignant']))






















