


import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
print(titanic.info())

y = titanic['survived']
X = titanic.drop(['row.names','name', 'survived'],axis=1)

# 填充空值
X['age'].fillna(X['age'].mean(), inplace=True)
X.fillna('UNKNOWN', inplace=True)

# 分割数据
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

# 类别型特征向量化
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='records'))
X_test = vec.transform((X_test.to_dict(orient='records')))
print(len(vec.feature_names_))

# 使用决策树模型依靠所有特征进行预测
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train, y_train)
print(dtc.score(X_test, y_test))

# 从sklearn导入特征筛选器
from sklearn import feature_selection
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)
X_train_fs = fs.fit_transform(X_train, y_train)
dtc.fit(X_train_fs, y_train)

X_test_fs = fs.transform(X_test)
print(dtc.score(X_test_fs, y_test))

from sklearn.model_selection import cross_val_score
import numpy as np 
percentiles = range(1,100,2)

results_train = []
results_test = []
for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=i)
    
    X_train_fs = fs.fit_transform(X_train, y_train)
    # results_train.append(dtc.score(X_train_fs, y_train))
    scores_train = cross_val_score(dtc, X_train_fs, y_train, cv=3)
    results_train.append(np.mean(scores_train))
    dtc.fit(X_train_fs, y_train)
    X_test_fs = fs.transform(X_test)
    results_test.append(dtc.score(X_test_fs, y_test))

# print(results)
# print(np.max(results))
# print(np.argmax(results))

# 画图展示
import matplotlib.pyplot as plt 
plt.plot(percentiles, results_train, 'b-')
plt.plot(percentiles, results_test, 'r--')
plt.xlabel('percentiles of features')
plt.ylabel('accuracy')
plt.show()




















