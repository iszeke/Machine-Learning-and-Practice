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
#


# 数据分割
import numpy as np
from sklearn.model_selection import train_test_split

X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
print(np.max(y))
print(np.min(y))
print(np.mean(y))
#

# 标准化处理
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
ss_y = StandardScaler()
X_train = ss_X.fit_transform(X_train)
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
X_test = ss_X.transform(X_test)
y_test = ss_y.transform(y_test.reshape(-1, 1))
#

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 线性回归
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_y_predict = lr.predict(X_test)

print('\n线性回归')
print('r2_score: ', r2_score(y_test, lr_y_predict))
print('mse: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))
print('mae: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))


# 径向基支持向量机
from sklearn.svm import SVR
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train.ravel())
svr_y_predict = svr.predict(X_test)

print('\n径向基支持向量机')
print('r2_score: ', r2_score(y_test, svr_y_predict))
print('mse: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(svr_y_predict)))
print('mae: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(svr_y_predict)))


# K近邻算法
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors=5, weights='distance')
knr.fit(X_train, y_train)
knr_y_predict = knr.predict(X_test)

print('\nK近邻回归算法')
print('r2_score: ', r2_score(y_test, knr_y_predict))
print('mse: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(knr_y_predict)))
print('mae: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(knr_y_predict)))


# 回归树算法
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=33)
dtr.fit(X_train, y_train)
dtr_y_predict = dtr.predict(X_test)

print('\n回归树算法')
print('r2_score: ', r2_score(y_test, dtr_y_predict))
print('mse: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict)))
print('mae: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict)))


# 随机森林
from sklearn.ensemble import RandomForestRegressor
rfg = RandomForestRegressor(n_estimators=200, random_state=33, max_features='sqrt')
rfg.fit(X_train, y_train.ravel())
rfg_y_predict = rfg.predict(X_test)

print('\n随机森林')
print('r2_score: ', r2_score(y_test, rfg_y_predict))
print('mse: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfg_y_predict)))
print('mae: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfg_y_predict)))
print(np.sort(list(zip(rfg.feature_importances_, boston.feature_names)),axis=0))

# 极端随机森林
from sklearn.ensemble import ExtraTreesRegressor
etr = ExtraTreesRegressor(n_estimators=500, random_state=33, max_features='sqrt')
etr.fit(X_train, y_train)
etr_y_predict = etr.predict(X_test)

print('\n极端随机森林')
print('r2_score: ', r2_score(y_test, etr_y_predict))
print('mse: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict)))
print('mae: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict)))


# 梯度提升树
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, subsample=0.8, random_state=33, max_features='sqrt')
gbr.fit(X_train, y_train)
gbr_y_predict = gbr.predict(X_test)

print('\n梯度提升树')
print('r2_score: ', r2_score(y_test, gbr_y_predict))
print('mse: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict)))
print('mae: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict)))