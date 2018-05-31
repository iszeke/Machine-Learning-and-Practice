# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     新闻分类
   Author :       Zeke
   date：          2018/5/31
   Description :   朴素贝叶斯
-------------------------------------------------
"""
# 导入数据
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')
print(len(news.data))
print(news.data[0])

# 分割数据
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.23, random_state=33)

# 特征抽取/拟合/验证
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# 不使用停用词CountVectorizer
cv = CountVectorizer()
cv_X_train = cv.fit_transform(X_train)
cv_X_test = cv.transform(X_test)

cv_mnb = MultinomialNB()
cv_mnb.fit(cv_X_train, y_train)
cv_y_predict = cv_mnb.predict(cv_X_test)
print('')
print('不使用停用词 cv accuracy: ', cv_mnb.score(cv_X_test, y_test))
print('不使用停用词 cv 分类报告: ', classification_report(y_test, cv_y_predict, target_names=news.target_names))

# 使用停用词CountVectorizer
cvf = CountVectorizer(analyzer='word', stop_words='english')
cvf_X_train = cvf.fit_transform(X_train)
cvf_X_test = cvf.transform(X_test)

cvf_mnb = MultinomialNB()
cvf_mnb.fit(cvf_X_train, y_train)
cvf_y_predict = cvf_mnb.predict(cvf_X_test)
print('')
print('使用停用词 cvf accuracy: ', cvf_mnb.score(cvf_X_test, y_test))
print('使用停用词 cvf 分类报告: ', classification_report(y_test, cvf_y_predict, target_names=news.target_names))

# 不使用停用词TfidfVectorizer
tv = TfidfVectorizer()
tv_X_train = tv.fit_transform(X_train)
tv_X_test = tv.transform(X_test)

tv_mnb = MultinomialNB()
tv_mnb.fit(tv_X_train, y_train)
tv_y_predict = tv_mnb.predict(tv_X_test)
print('')
print('不使用停用词 tv accuracy: ', tv_mnb.score(tv_X_test, y_test))
print('不使用停用词 tv 分类报告: ', classification_report(y_test, tv_y_predict, target_names=news.target_names))

# 使用停用词TfidfVectorizer
tvf = TfidfVectorizer(analyzer='word', stop_words='english')
tvf_X_train = tvf.fit_transform(X_train)
tvf_X_test = tvf.transform(X_test)

tvf_mnb = MultinomialNB()
tvf_mnb.fit(tvf_X_train, y_train)
tvf_y_predict = tvf_mnb.predict(tvf_X_test)
print('')
print('使用停用词 tvf accuracy: ', tvf_mnb.score(tvf_X_test, y_test))
print('使用停用词 tvf 分类报告: ', classification_report(y_test, tvf_y_predict, target_names=news.target_names))






