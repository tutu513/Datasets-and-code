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
data = pd.read_csv("上海预处理后的数据归一化数据.csv")
data = data.drop(['Unnamed: 0'], axis = 1)
data.head()
#%%
n, m = len(data), len(data.T)
data1 = np.zeros((n,m))
t = 0
for i in range(0,n):
    a = 1
    for j in range(0,m):
        a = a + data.loc[i][j]
        data1[t][j] = a
    t = t + 1
data1 = pd.DataFrame(data1)
data1.head()
#%%
df = pd.read_excel("上海数据K-means结果.xlsx")
df.head()
#%%
a1, a2, a3 = [], [], []
for j in range(0,len(data)):
    if df.loc[j][1] == 1:
        a1.append(j)
    elif df.loc[j][1] == 2:
        a2.append(j)
    else:
        a3.append(j)
print(a1)
print(a2)
print(a3)
#%%

#%%

#%%
# 我们模型的结果
lb1 = [0, 6, 9, 13, 19, 25, 26]
lb2 = [4, 21]
lb3 = [1, 2, 3, 5, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18, 20, 22, 23, 24, 27, 28, 29]

## K-means聚类结果
A1 = [3, 6, 7, 13, 17]
A2 = [0, 1, 2, 5, 11, 12, 14, 18, 19, 20, 22, 23, 24, 27, 28, 29]
A3 = [4, 8, 9, 10, 15, 16, 21, 25, 26]

# 三支聚类结果
B1 = [0, 3, 12, 16, 17, 24, 25, 27]
B2 = [4, 5, 7, 11, 13, 14, 15, 20, 22, 28, 29]
B3 = [1, 2, 6, 8, 9, 10, 18, 19, 21, 23, 26]

# 基于遗憾理论的灰度关联聚类结果
C1 = [2, 3, 5, 6, 7, 11, 12, 13, 14, 15, 17, 19, 22, 23, 25, 26, 27, 28]
C2 = [0, 1, 4, 8, 9, 10, 16, 18, 20, 21, 24, 29]

# 亲和传播聚类结果
D1 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
D2 = [9, 10, 11, 12, 13, 14, 15, 16, 17]
D3 = [18, 19, 20, 21, 22, 23]
D4 = [24, 25, 26, 27, 28, 29]
#%%
# 每个股票的期望收益率
ExpReturn = []

for i in range(0,n):
    ExpReturn.append(np.mean(data1.iloc[i]))
    ExpReturn[i] = round(ExpReturn[i], 3)
print(ExpReturn)
#%%
# 每个股票的协方差矩阵
ExpCovar = np.cov(data1)
for i in range(0,n):
    for j in range(0,n):
        ExpCovar[i][j] = round(ExpCovar[i][j], 3)
ExpCovar = pd.DataFrame(ExpCovar)
ExpCovar.head()
#%%
ExpCovar.to_csv(r"上海股票的协方差矩阵.csv")
#%%
w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
t = 0
t1 = 0
for l in range(0,len(w)):
    for i1 in range(0,n):
        if i1 in lb1:
            for i2 in range(0,n):
                if i2 in lb3:
                    Y = w[l]*data.iloc[i1]+(1-w[l])*data.iloc[i2]
                    s = pd.Series(Y)
                    if (np.mean(Y)-0.3)/np.std(Y) > t:
                        t = (np.mean(Y)-0.3)/np.std(Y)   
                        t1 = np.std(Y)
print(t)
print(t1)
#%%

#%%

#%%
w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
t = 0
t1 = 0
for l in range(0,len(w)):
    for i1 in range(0,n):
        if i1 in A1:
            for i2 in range(0,n):
                if i2 in A3:
                    Y = w[l]*data.iloc[i1]+(1-w[l])*data.iloc[i2]
                    s = pd.Series(Y)
                    if (np.mean(Y)-0.3)/np.std(Y) > t:
                        t = (np.mean(Y)-0.3)/np.std(Y)   
                        t1 = np.std(Y)
print(t)
print(t1)
#%%

#%%
w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
t = 0
t1 = 0
for l in range(0,len(w)):
    for i1 in range(0,n):
        if i1 in B3:
            for i2 in range(0,n):
                if i2 in B2:
                    Y = w[l]*data.iloc[i1]+(1-w[l])*data.iloc[i2]
                    s = pd.Series(Y)
                    if (np.mean(Y)-0.3)/np.std(Y) > t:
                        t = (np.mean(Y)-0.3)/np.std(Y)   
                        t1 = np.std(Y)
print(t)
print(t1)
#%%
w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
t = 0
t1 = 0
for l in range(0,len(w)):
    for i1 in range(0,n):
        if i1 in C1:
            for i2 in range(0,n):
                if i2 in C2:
                    Y = w[l]*data.iloc[i1]+(1-w[l])*data.iloc[i2]
                    s = pd.Series(Y)

                    t = (np.mean(Y)-0.3)/np.std(Y)
                    t1 = np.std(Y)
print(t)
print(t1)
#%%
w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
t = 0
t1 = 0
for l in range(0,len(w)):
    for i1 in range(0,n):
        if i1 in D1:
            for i2 in range(0,n):
                if i2 in D2:
                    Y = w[l]*data.iloc[i1]+(1-w[l])*data.iloc[i2]
                    s = pd.Series(Y)
                    if (np.mean(Y)-0.3)/np.std(Y) > t:
                        t = (np.mean(Y)-0.3)/np.std(Y)   
                        t1 = np.std(Y)
print(t)
print(t1)
#%%
config = {
            "font.family": 'serif',
            "font.size": 14,# 相当于小四大小
            "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            "font.serif": ['SimSun'],#宋体
            'axes.unicode_minus': False # 处理负号，即-号
         }
rcParams.update(config)
#%%
#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
rc('font',size=16); rc('text', usetex=True)  #调用tex字库

plt.rcParams['figure.figsize'] = (13.0, 5)
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18,
         }
# 按两行两咧显示
X = ['I', 'II', 'III', 'IV', 'V']
Y1 = [3.351128199924945, 2.996106287411055, 2.1328322218767726, 2.9719382029005064, 3.1720297640238218]
Y2 =[0.10238110992419902, 0.1069955267082206, 0.08696080385880804, 0.11588258112453977, 0.10325337077503187]
a = subplot(1,2,1) #在第一窗口显示
bar_width = 0.3 # 条形宽度

# 使用两bar 函数画出两组条形图
plt.bar(X, height=Y1, width=bar_width, color='royalblue')
plt.xlabel('Model',font1)
plt.ylabel('Sharpe Ratio',font1)

b = subplot(1,2,2) 
bar_width = 0.3 # 条形宽度
index = np.arange(len(X)) # 条形图的横坐标

# 使用两bar 函数画出两组条形图
plt.bar(X, height=Y2, width=bar_width, color='coral')
plt.xlabel('Model',font1)
plt.ylabel('Volatility',font1)
# savefig("上海数据投资分析图.jpg",dpi=300, bbox_inches='tight')
show()
#%%

#%%
