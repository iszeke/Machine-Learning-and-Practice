

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

# 画第一个图
plt.subplot(3, 2, 1)
plt.scatter(x1, x2)
plt.xlim([0, 10])
plt.ylim([0, 10])

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']

clusters = [2, 3, 4, 5, 8]
subplot_counter = 1
sc_scores = []
meandistortions = []
for t in clusters:
    subplot_counter += 1
    plt.subplot(3, 2, subplot_counter)

    kmeans_model = KMeans(n_clusters=t).fit(X)

    # 画出聚类中心点
    for i, j in enumerate(kmeans_model.labels_):
        plt.plot(x1[i], x2[i], color=colors[j], marker=markers[j], ls='None')
        plt.xlim([0, 10])
        plt.ylim([0, 10])

    sc_score = silhouette_score(X, kmeans_model.labels_, metric='euclidean')
    sc_scores.append(sc_score)

    print(kmeans_model.cluster_centers_.shape)
    meandistortions.append(np.sum(np.min(cdist(X, kmeans_model.cluster_centers_),axis=1))/X.shape[0])
    print(cdist(X, kmeans_model.cluster_centers_).shape)
    print(np.min(cdist(X, kmeans_model.cluster_centers_),axis=1).shape)
    plt.title('K=%s, silhouette coefficient=%0.03f' %(t, sc_score))
plt.show()

# 绘制轮廓系数与簇数量的曲线
plt.figure()
plt.plot(clusters, sc_scores, '*-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient Score')
plt.show()

# 肘部观察法
plt.figure()
plt.plot(clusters, meandistortions, '*-')
plt.xlabel('Number of Clusters')
plt.ylabel('Average Dispersion')
plt.show()

































