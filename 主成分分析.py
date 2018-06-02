# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     主成分分析
   Author :       Zeke
   date：          2018/6/2
   Description :   查看主成分分析对手写字体识别性能的变化
-------------------------------------------------
"""

import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report

class DigitsRecognition():
    "比较使用pca与不使用pca的结果差异"
    def __init__(self):
        pass

    def data_process(self):
        # 读取数据集
        os.chdir(r'C:\Users\Zeke\Downloads')
        digits_train = pd.read_csv('optdigits.tra', header=None)
        digits_test = pd.read_csv('optdigits.tes', header=None)
        # digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
        # digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',header=None)
        print(digits_train.shape)
        print(digits_test.shape)

        # 分割X与y
        self.X_train = digits_train.iloc[:, :-1]
        self.y_train = digits_train.iloc[:, -1]
        self.X_test = digits_test.iloc[:, :-1]
        self.y_test = digits_test.iloc[:, -1]

    def dim2_pca_show(self):
        "展示将维度压缩到2维时的图片"
        # 初始化
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(self.X_train)

        # 显示10类手写字体经pca压缩后的2维空间分布
        color = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
        for i in range(10):
            px = X_train_pca[:, 0][self.y_train == i]
            py = X_train_pca[:, 1][self.y_train == i]
            plt.scatter(px, py, c=color[i])
        plt.legend(np.arange(0, 10).astype(str))
        plt.xlabel('First Principal Component')
        plt.xlabel('Second Principal Component')
        plt.show()

    def score_of_no_pca(self):
        "不使用pca，查看支持向量机的性能"
        # 支持向量机
        svc = SVC(kernel='linear')
        svc.fit(self.X_train, self.y_train)
        y_predict = svc.predict(self.X_test)
        # 性能
        print(svc.score(self.X_test, self.y_test))
        print(classification_report(self.y_test, y_predict, target_names=np.arange(10).astype(str)))

    def score_of_yes_pca(self, n_components):
        "使用pca，查看支持向量机的性能"
        # pca
        pca = PCA(n_components=20)
        X_train_pca = pca.fit_transform(self.X_train)
        # 支持向量机
        pca_svc = SVC(kernel='linear')
        pca_svc.fit(X_train_pca, self.y_train)
        pca_y_predict = pca_svc.predict(pca.transform(self.X_test))
        # 性能
        print(pca_svc.score(pca.transform(self.X_test), self.y_test))
        print(classification_report(self.y_test, pca_y_predict, target_names=np.arange(10).astype(str)))

if __name__ == '__main__':
    dr = DigitsRecognition()
    dr.data_process()
    dr.dim2_pca_show()
    dr.score_of_no_pca()
    dr.score_of_yes_pca(10)

