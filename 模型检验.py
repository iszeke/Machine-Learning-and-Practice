
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


if __name__ == '__main__':

    news = fetch_20newsgroups(subset='all')
    X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

    # 使用Pipeline简化系统搭建
    clf = Pipeline([('vect',TfidfVectorizer(stop_words='english', analyzer='word')),
                    ('svc', SVC())])

    # 实验的svc超参数(gamma, C)
    paras = {'svc__gamma':np.logspace(-2, 1, 4),
             'svc__C':np.logspace(-1, 1, 3)}

    gs = GridSearchCV(clf, paras, verbose=2, refit=True, cv=3, n_jobs=-1) # n_jobs是并行计算

    # 执行单线程搜索
    gs.fit(X_train, y_train)

    print(gs.best_params_, gs.best_score_)

    # 输出最佳模型再测试集上的准确性
    print(gs.score(X_test, y_test))


















