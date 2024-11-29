#%%
## 导入需要的库
import numpy as np
import pandas as pd
import math
import os
import random
from matplotlib.pyplot import *
import matplotlib.pyplot as plt

from statsmodels.stats.weightstats import ztest
#%%
## 我们模型的聚类结果如下所示
lb1 = [1, 10, 13, 18, 22, 24, 25, 28, 29]
lb2 = [2, 5, 7, 8, 19, 26]
lb3 = [0, 3, 4, 6, 9, 11, 12, 14, 15, 16, 17, 20, 21, 23, 27]
label1 = []
for i in range(0,30):
    if i in lb1:
        label1.append(1)
    elif i in lb2:
        label1.append(0)
    else:
        label1.append(2)
print(label1)
#%%
## K-means聚类结果
from sklearn.cluster import KMeans
X = pd.read_csv("上海预处理后的数据归一化数据.csv")
estimator = KMeans(n_clusters = 3)#构造聚类器
estimator.fit(X)#聚类
label2 = estimator.labels_ #获取聚类标签
label2
#%%
a1, a2, a3 = [], [], []
for j in range(0,len(X)):
    if label2[j] == 2:
        a1.append(j)
    elif label2[j] == 1:
        a2.append(j)
    else:
        a3.append(j)
print(a1)
print(a2)
print(a3)
#%%
## 三支聚类
data = X.drop(['Unnamed: 0'], axis = 1)
data.head()
#%%
### 设定正理想解和负理想解
n, m = len(data), len(data.T)

### 计算决策对象到正负理想解的距离
D1 = np.zeros((1,n));D0 = np.zeros((1,n)); Pr = []
for i in range(0,n):
    D1[0][i] = np.sqrt(np.sum(np.square(1 - data.iloc[i]))) 
    D0[0][i] = np.sqrt(np.sum(np.square(data.iloc[i])))
    Pr.append(D0[0][i]/(D0[0][i] + D1[0][i]))
print(Pr)
#%%
def sunshi(x,singam):
    lamda = np.zeros((len(x),6));
    juhe = np.zeros((1,6));
    
    # 计算相对损失函数
    for i in range(0,len(x)):
        lamda[i,1] = singam[i] * x[i];
        lamda[i,2] = x[i];
        lamda[i,3] = 1-x[i];
        lamda[i,4] = singam[i] * (1-x[i]);
    
    # 计算聚合相对效用函数(均值聚合)
    juhe = np.mean(lamda,axis=0)
    
    return juhe
#%%
# 计算不同对象的聚合相对损失函数矩阵
singam = [0.45]*m
lamda = np.zeros((n,6));
for i in range(0,n):
    x = data.iloc[i]
    lamda[i] = sunshi(x ,singam)

#根据聚合相对损失函数计算不同对象的初始阈值
yuzhi = np.zeros((n,2))
for i in range(0,n):
    for j in range(0,len(lamda.T)):
        yuzhi[i][0] = (lamda[i][3]-lamda[i][4])/(lamda[i][3]-lamda[i][4]+lamda[i][1])
        yuzhi[i][1] = (lamda[i][4])/(lamda[i][4]+lamda[i][2]-lamda[i][1])
yuzhi = pd.DataFrame(yuzhi)
yuzhi.head()
#%%
B1, B2, B3 = [], [], []
for i in range(n):
    if Pr[i] > yuzhi.loc[i][0]:
        B1.append(i)
    elif Pr[i]> yuzhi.loc[i][1] and Pr[i] < yuzhi.loc[i][0]:
        B2.append(i)
    else:
        B3.append(i)
print(B1)
print(B2)
print(B3)
#%%
label3 = []
for i in range(0,30):
    if i in B1:
        label3.append(0)
    elif i in B2:
        label3.append(1)
    else:
        label3.append(2)
print(label3)
#%%
## 基于遗憾理论的灰度关联聚类方法
r1 = np.zeros((n,m))
r2 = np.zeros((n,m))

for i in range(n):
    for j in range(m):
        r1[i][j] = 1/(1.5-data.loc[i][j])
        r2[i][j] = 1/(data.loc[i][j]+0.5)
#%%
# 建立感知效用矩阵
def Juzhen(alpha, sigma):
    U = np.zeros((n,m))
    R = r1/(r1+r2)
    G = 1 - np.exp(sigma * abs(r1-1))
    Q = 1 - np.exp(-sigma * abs(r1-1))
    for i in range(n):
        for j in range(m):
            U[i][j] = (R[i][j]) ** alpha + G[i][j] + Q[i][j] 
    return U
#%%
U = Juzhen(0.5,0.15)
U
#%%
# 计算聚类系数
u = []
w = [0.1,0.2,0.5,0.05,0.15]
for i in range(n):
    u.append(np.sum(w * U[i][m-5:]))
print(u)
#%%
C1, C2 = [], []
for i in range(n):
    if u[i] > np.mean(u):
        C1.append(i)
    else:
        C2.append(i)
print(C1)
print(C2)
#%%
from numpy import unique
from sklearn.cluster import AffinityPropagation
import numpy as np
from sklearn.preprocessing import StandardScaler

model = AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None,    
                            affinity='euclidean', verbose=False) # 设置damping : 阻尼系数，取值[0.5,1)
# 匹配模型
model.fit(X)
yhat = model.predict(X)  # yhat为集群结果
clusters = len(unique(yhat))  # 类别

print(yhat)
#%%
D1, D2, D3, D4 = [], [], [], []
for i in range(n):
    if yhat[i] == 0:
        D1.append(i)
    if yhat[i] == 1:
        D2.append(i)
    if yhat[i] == 2:
        D3.append(i)
    if yhat[i] == 3:
        D4.append(i)
print(D1)
print(D2)
print(D3)
print(D4)
#%%
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
#%%
# 计算Silhouette系数
silhouette_avg = silhouette_score(X, label2)
print(f"Silhouette Coefficient: {silhouette_avg}")

# 计算Davies-Bouldin指数
dbi = davies_bouldin_score(X, label2)
print(f"Davies-Bouldin Index: {dbi}")
#%%
# 计算Silhouette系数
silhouette_avg = silhouette_score(X, label3)
print(f"Silhouette Coefficient: {silhouette_avg}")

# 计算Davies-Bouldin指数
dbi = davies_bouldin_score(X, label3)
print(f"Davies-Bouldin Index: {dbi}")
#%%
