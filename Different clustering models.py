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
#%% md
# # 聚类结果比较
#%% md
# ## 我们模型的聚类
#%%
## 聚类结果如下所示
lb1 = [20, 1, 4, 5, 6, 9, 16, 22, 25, 34, 36, 38, 43, 44, 46, 49]
lb2 = [12, 17, 27, 2, 3, 8, 10, 21, 23, 24, 26, 29, 30, 31, 33, 37, 47, 48]
lb3 = [0, 7, 11, 13, 14, 15, 18, 19, 28, 32, 35, 39, 40, 41, 42, 45]
#%%

#%%

#%%

#%% md
# ## K-means聚类
#%%
from sklearn.cluster import KMeans
X = data
estimator = KMeans(n_clusters = 4)#构造聚类器
estimator.fit(X)#聚类
label_pred = estimator.labels_ #获取聚类标签
label_pred
#%%
a1, a2, a3 = [], [], []

for j in range(0,len(df1)):
    if label_pred[j] == 1:
        a1.append(j)
    elif label_pred[j] == 3:
        a2.append(j)
    else:
        a3.append(j)
print(a1)
print(a2)
print(a3)
#%%
# 读取结果
df1 = pd.read_excel("K-means聚类结果.xlsx")
df1 = df1.drop(['Unnamed: 0'], axis = 1)
df1.head()
#%%
A1, A2, A3 = [], [], []

for j in range(0,len(df1)):
    if df1['聚类'][j] == 1:
        A1.append(j)
    elif df1['聚类'][j] == 2:
        A2.append(j)
    else:
        A3.append(j)
print(A1)
print(A2)
print(A3)
#%%
## 根据K-means的距离对分类结果进行降序排序
AA1 = sorted(A1, key=lambda x: df1['距离'][A1.index(x)], reverse = True)
AA2 = sorted(A2, key=lambda x: df1['距离'][A2.index(x)], reverse = True)
AA3 = sorted(A3, key=lambda x: df1['距离'][A3.index(x)], reverse = True)
print(AA1)
print(AA2)
print(AA3)
#%%

#%% md
# ## 三支聚类
#%%
data = pd.read_csv("归一化数据.csv")
data = data.drop(['Unnamed: 0'], axis = 1)
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
    elif Pr[i]> 0.52 and Pr[i] < yuzhi.loc[i][0]:
        B2.append(i)
    else:
        B3.append(i)
print(B1)
print(B2)
print(B3)
#%% md
# ## 基于遗憾理论的灰度关联聚类方法
#%%
data.head()
#%%
r1 = np.zeros((n,m))
r2 = np.zeros((n,m))

for i in range(n):
    for j in range(m):
        r1[i][j] = 1/(1.5-data.loc[i][j])
        r2[i][j] = 1/(data.loc[i][j]+0.5)
print(r1)
#%%
print(r2)
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
#%% md
# # 亲和传播聚类算法
#%%
from numpy import unique
from sklearn.cluster import AffinityPropagation
import numpy as np
from sklearn.preprocessing import StandardScaler

# 定义数据集
X = data
X = StandardScaler().fit_transform(X)
# 定义模型
model = AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None,    
                            affinity='euclidean', verbose=False) # 设置damping : 阻尼系数，取值[0.5,1)
# 匹配模型
model.fit(X)
yhat = model.predict(X)  # yhat为集群结果
clusters = len(unique(yhat))  # 类别

print(yhat)
#%%
D1, D2 = [], []
for i in range(n):
    if yhat[i] == 2:
        D1.append(i)
    else:
        D2.append(i)
print(D1)
print(D2)
#%% md
# ## 逆协方差聚类
#%%
# 计算协方差
V = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        V[i][j] = np.min(np.corrcoef(data.iloc[i],data.iloc[j]))
# 计算逆协方差
V1 = np.linalg.inv(V)
V1
#%%
V2 = pd.DataFrame(V1)
V2.head()
#%%
l = np.random.randint(0,49)
print(l)
E1, E2 = [], []
for i in range(n):
    if V2.loc[l][i]>0:
        E1.append(i)
    else:
        E2.append(i)
print(E1)
print(E2)
#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%
def getEuclidean(point1, point2):
    dimension = len(point1)
    dist = 0.0
    for i in range(dimension):
        dist += (point1[i] - point2[i]) ** 2
    return math.sqrt(dist)
#%%
dist = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        dist[i][j] = getEuclidean(data.iloc[i], data.iloc[j])
dist
#%%

#%%
