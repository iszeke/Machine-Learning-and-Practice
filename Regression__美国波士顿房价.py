# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     波士顿房价预测
   Author :       Zeke
   date：          2018/5/30
   Description :   多元线性回归
-------------------------------------------------
"""

# 导入波士顿房价数据
from sklearn.datasets import load_boston
boston = load_boston()
# print(boston.DESCR)


# 数据分割
import numpy as np
from sklearn.model_selection import train_test_split

X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)
print(np.max(y))
print(np.min(y))
print(np.mean(y))

# 标准化处理
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
ss_y = StandardScaler()
X_train = ss_X.fit_transform(X_train)
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
X_test = ss_X.transform(X_test)
y_test = ss_y.transform(y_test.reshape(-1, 1))

# 使用线性回归模型对波士顿房价进行预测
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_y_predict = lr.predict(X_test)

# 性能评测
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print('r2_score: ', r2_score(y_test, lr_y_predict))
print('mse: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))
print('mae: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))













