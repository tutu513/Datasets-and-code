#%%
## 导入需要的库
import numpy as np
import pandas as pd
import math
import os
import random
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
#%%
## 我们模型的聚类结果如下所示
lb1 = [20, 1, 4, 5, 6, 9, 16, 22, 25, 34, 36, 38, 43, 44, 46, 49]
lb2 = [12, 17, 27, 2, 3, 8, 10, 21, 23, 24, 26, 29, 30, 31, 33, 37, 47, 48]
lb3 = [0, 7, 11, 13, 14, 15, 18, 19, 28, 32, 35, 39, 40, 41, 42, 45]

## K-means聚类结果
A1 = [4, 5, 7, 14, 15, 18, 31, 32, 36, 37, 41, 45]
A2 = [0, 2, 3, 6, 8, 9, 10, 11, 12, 13, 16, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 38, 39, 42, 43, 44, 46, 47, 48, 49]
A3 = [1, 17, 19, 21, 40]

# 三支聚类结果
B1 = [0, 5, 9, 10, 19, 23, 26, 29, 30, 32, 35, 36, 41, 43, 45, 48]
B2 = [2, 3, 4, 8, 11, 12, 13, 14, 15, 17, 18, 20, 24, 27, 31, 33, 34, 37, 38, 39, 40, 42, 44, 46, 47, 49]
B3 = [1, 6, 7, 16, 21, 22, 25, 28]

# 基于遗憾理论的灰度关联聚类结果
C1 = [2, 3, 6, 9, 11, 18, 19, 21, 23, 30, 32, 33, 35, 37, 38, 41, 46]
C2 = [0, 1, 4, 5, 7, 8, 10, 12, 13, 14, 15, 16, 17, 20, 22, 24, 25, 26, 27, 28, 29, 31, 34, 36, 39, 40, 42, 43, 44, 45, 47, 48, 49]

# 亲和传播聚类结果
D1 = [0, 1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
D2 = [2, 6, 21, 26]
#%%
def getEuclidean(point1, point2):
    dimension = len(point1)
    dist = 0.0
    for i in range(dimension):
        dist += (point1[i] - point2[i]) ** 2
    return math.sqrt(dist)
#%%
## 读取数据
data = pd.read_csv("归一化数据.csv")
data = data.drop(['Unnamed: 0'], axis = 1)
data.head()
#%%
n = len(data)
dist = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        dist[i][j] = getEuclidean(data.iloc[i][1490:], data.iloc[j][1490:])
#%%
dist1 = np.zeros((n,n))
for i in range(n):
    a = min(dist[i])
    b = max(dist[i])
    for j in range(n):
        dist1[i][j]=(dist[i][j]-a)/(b-a)
dist1
#%%
labels1 = [] 
for i in range(n):
    if i in lb1:
        labels1.append(0)
    elif i in lb2:
        labels1.append(1)
    else:
        labels1.append(2)
print(labels1)
#%%
labels2 = [] 
for i in range(n):
    if i in A1:
        labels2.append(0)
    elif i in A2:
        labels2.append(1)
    else:
        labels2.append(2)
print(labels2)
#%%
labels3 = [] 
for i in range(n):
    if i in B1:
        labels3.append(0)
    elif i in B2:
        labels3.append(1)
    else:
        labels3.append(2)
print(labels3)
#%%
labels4 = [] 
for i in range(n):
    if i in C1:
        labels4.append(0)
    else:
        labels4.append(1)
print(labels4)
#%%
labels5 = [] 
for i in range(n):
    if i in D1:
        labels5.append(0)
    else:
        labels5.append(1)
print(labels5)
#%%
# 计算轮廓系数
from sklearn.metrics import silhouette_score
silhouette_avg1 = silhouette_score(dist1, labels1)
silhouette_avg2 = silhouette_score(dist1, labels2)
silhouette_avg3 = silhouette_score(dist1, labels3)
silhouette_avg4 = silhouette_score(dist1, labels4)
silhouette_avg5 = silhouette_score(dist1, labels5)
print("轮廓系数:", silhouette_avg1)
print("轮廓系数:", silhouette_avg2)
print("轮廓系数:", silhouette_avg3)
print("轮廓系数:", silhouette_avg4)
print("轮廓系数:", silhouette_avg5)
#%%
from sklearn.metrics import calinski_harabaz_score
score1 = calinski_harabaz_score(dist1, labels1)
score2 = calinski_harabaz_score(dist1, labels2)
score3 = calinski_harabaz_score(dist1, labels3)
score4 = calinski_harabaz_score(dist1, labels4)
score5 = calinski_harabaz_score(dist1, labels5)
print("Calinski-Harabasz指数:", score1)
print("Calinski-Harabasz指数:", score2)
print("Calinski-Harabasz指数:", score3)
print("Calinski-Harabasz指数:", score4)
print("Calinski-Harabasz指数:", score5)
#%%
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.datasets import make_blobs

# 生成聚类数据
X, _ = make_blobs(n_samples=50, n_features=1500)

# 定义评价指标列表
silhouette_scores = []
calinski_harabasz_scores = []
davies_bouldin_scores = []
dunn_scores = []

from sklearn.metrics import pairwise_distances
import numpy as np

def calculate_dunn_index(X, labels):
    """
    计算 Dunn 指数：最小簇间距离除以最大簇内距离的比值
    参数：
      - X：样本数据矩阵，形状为 (n_samples, n_features)
      - labels：聚类结果标签，形状为 (n_samples,)
    返回：
      - dunn_index：Dunn 指数
    """

    # 计算距离矩阵
    distances = pairwise_distances(X)

    # 根据聚类结果计算簇内距离和簇间距离
    cluster_distances = []
    for i in np.unique(labels):
        cluster_i_indices = np.where(labels == i)[0]
        cluster_i_distances = distances[cluster_i_indices][:, cluster_i_indices]
        cluster_distances.append(cluster_i_distances)

    cluster_distances = np.concatenate(cluster_distances)

    # 计算最小簇间距离
    min_intercluster_distance = np.min(cluster_distances[cluster_distances > 0])

    # 计算最大簇内距离
    max_intracluster_distance = np.max(np.array([np.max(cluster_distance) for cluster_distance in cluster_distances]))

    # 计算 Dunn 指数
    dunn_index = min_intercluster_distance / max_intracluster_distance

    return dunn_index

# 随机生成五个聚类模型并计算评价指标
for _ in range(5):
    # 聚类算法
    kmeans = KMeans(n_clusters=5)
    agg = AgglomerativeClustering(n_clusters=5)
    labels_kmeans = kmeans.fit_predict(X)
    labels_agg = agg.fit_predict(X)

    # 计算轮廓系数
    silhouette_kmeans = silhouette_score(X, labels_kmeans)
    silhouette_agg = silhouette_score(X, labels_agg)
    silhouette_scores.append([silhouette_kmeans, silhouette_agg])

    # 计算Calinski-Harabasz指数
    calinski_harabasz_kmeans = calinski_harabasz_score(X, labels_kmeans)
    calinski_harabasz_agg = calinski_harabasz_score(X, labels_agg)
    calinski_harabasz_scores.append([calinski_harabasz_kmeans, calinski_harabasz_agg])

    # 计算Davies-Bouldin指数
    davies_bouldin_kmeans = davies_bouldin_score(X, labels_kmeans)
    davies_bouldin_agg = davies_bouldin_score(X, labels_agg)
    davies_bouldin_scores.append([davies_bouldin_kmeans, davies_bouldin_agg])

    # 计算Dunn指数
    dunn_kmeans = calculate_dunn_index(X, labels_kmeans)
    dunn_agg = calculate_dunn_index(X, labels_agg)
    dunn_scores.append([dunn_kmeans, dunn_agg])

# 将四个评价指标转化为越大越好的形式，即通过取相反数实现
silhouette_scores = [[-score[0], -score[1]] for score in silhouette_scores]
calinski_harabasz_scores = [[-score[0], -score[1]] for score in calinski_harabasz_scores]
davies_bouldin_scores = [[-score[0], -score[1]] for score in davies_bouldin_scores]
dunn_scores = [[-score[0], -score[1]] for score in dunn_scores]

# 输出评价指标结果
print("轮廓系数：", silhouette_scores)
print("Calinski-Harabasz指数：", calinski_harabasz_scores)
print("Davies-Bouldin指数：", davies_bouldin_scores)
print("Dunn指数：", dunn_scores)
#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%
X = ('Model I', 'Model II', 'Model III', 'Model IV', 'Model V') 
# X = ('I', 'II', 'III', 'IV', 'V') 
Y1 = [0.71, 0.55, 0.63, 0.47, 0.51]
Y2 = [8.35, 7.75, 7.54, 6.72, 6.08]
Y3 = [0.27, 0.40, 0.35, 0.45, 0.48]
Y4 = [7.55, 6.56, 6.44, 5.78, 5.46]
plt.rcParams['figure.figsize'] = (12.0, 10.0)
#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
rc('font',size=16); rc('text', usetex=True)  #调用tex字库

means = [5, 15, 25, 35, 45]
colormap = plt.get_cmap('coolwarm')
colors = colormap(np.linspace(0.1, 1, len(means)))

# 按两行三咧显示
a = subplot(2,2,1) #在第一窗口显示
a.bar(X,Y1, width=0.4, color = colors[0], label = '$SC$')
plt.title("$SC$") 
b = subplot(2,2,2) #在第二个窗口显示
b.bar(X,Y2, width=0.4, color = colors[1], label = '$CH$' )
plt.title("$CH$") 
# plt.legend()
c = subplot(2,2,3)
c.bar(X,Y3, width=0.4, color = colors[2], label = '$DBI$')
plt.title("$DBI$") 
d = subplot(2,2,4) 
d.bar(X,Y4, width=0.4, color = colors[3], label = '$DI$')
plt.title("$DI$")
savefig("聚类比较1.jpg", dpi=300, bbox_inches='tight')
show()
#%%
