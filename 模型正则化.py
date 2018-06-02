# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     模型正则化
   Author :       Zeke
   date：          2018/6/2
   Description :   查看模型正则化对性能的影响
-------------------------------------------------
"""
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge


class Regular():
    "正则化的意义展示"
    def __init__(self):
        self.X_train = [[6], [8], [10], [14], [18]]
        self.y_train = [[7], [9], [13], [17.5], [18]]
        self.X_test = [[6], [8], [11], [16]]
        self.y_test = [[8], [12], [15], [18]]
        self.xx = np.linspace(0, 26, 100).reshape(-1,1) # 用于测试的样本

    def lr_regress_train_nd(self, *args):
        "训练集表现"
        for i in args:
            poly_n = PolynomialFeatures(degree=i)
            X_train_poly_n = poly_n.fit_transform(self.X_train)
            xx_poly_n = poly_n.transform(self.xx)

            lr = LinearRegression()
            lr.fit(X_train_poly_n, self.y_train)
            yy = lr.predict(xx_poly_n)
            # 画图
            plt.plot(self.xx, yy, label=str(i))
        plt.scatter(self.X_train, self.y_train)
        plt.xlabel('Diameter pf Pizza')
        plt.ylabel('Price of Pizza')
        plt.title('Train')
        plt.legend()
        plt.show()

    def lr_regress_test_nd(self, *args):
        "训练集表现"
        for i in args:
            poly_n = PolynomialFeatures(degree=i)
            X_train_poly_n = poly_n.fit_transform(self.X_train)
            X_test_poly_n = poly_n.transform(self.X_test)

            lr = LinearRegression()
            lr.fit(X_train_poly_n, self.y_train)
            y_test_pred = lr.predict(X_test_poly_n)
            score = lr.score(X_test_poly_n, self.y_test)
            # 画图
            plt.plot(self.X_test, y_test_pred, label=str(i) + '__' +str(score))
        plt.scatter(self.X_test, self.y_test)
        plt.xlabel('Diameter pf Pizza')
        plt.ylabel('Price of Pizza')
        plt.title('Test with LR')
        plt.legend()
        plt.show()

    def lasso_regress_test_nd(self, *args):
        "训练集表现"
        for i in args:
            poly_n = PolynomialFeatures(degree=i)
            X_train_poly_n = poly_n.fit_transform(self.X_train)
            X_test_poly_n = poly_n.transform(self.X_test)

            lasso = Lasso(alpha=1)
            lasso.fit(X_train_poly_n, self.y_train)
            y_test_pred = lasso.predict(X_test_poly_n)
            score = lasso.score(X_test_poly_n, self.y_test)
            # 画图
            plt.plot(self.X_test, y_test_pred, label=str(i)+'__'+str(score))
        plt.scatter(self.X_test, self.y_test)
        plt.xlabel('Diameter pf Pizza')
        plt.ylabel('Price of Pizza')
        plt.title('Test with Lasso')
        plt.legend()
        plt.show()

    def ridge_regress_test_nd(self, *args):
        "训练集表现"
        for i in args:
            poly_n = PolynomialFeatures(degree=i)
            X_train_poly_n = poly_n.fit_transform(self.X_train)
            X_test_poly_n = poly_n.transform(self.X_test)

            ridge = Ridge(alpha=1)
            ridge.fit(X_train_poly_n, self.y_train)
            y_test_pred = ridge.predict(X_test_poly_n)
            score = ridge.score(X_test_poly_n, self.y_test)
            # 画图
            plt.plot(self.X_test, y_test_pred, label=str(i)+'__'+str(score))
        plt.scatter(self.X_test, self.y_test)
        plt.xlabel('Diameter pf Pizza')
        plt.ylabel('Price of Pizza')
        plt.title('Test with Ridge')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    a = Regular()
    a.lr_regress_train_nd(1,2,3)
    a.lr_regress_test_nd(1,2,3)
    a.lasso_regress_test_nd(4,5,6)
    a.ridge_regress_test_nd(4,5,6)









































