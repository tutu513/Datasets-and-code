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
## 读取数据
df = pd.read_excel("2022年上海证券交易所50指数（上证50指数）数据集/预处理后的数据.xlsx")
#%%
df.head()
#%%
#根据日收盘价计算日收益率，再计算日对数收益率
df1 = np.zeros((len(df),len(df.T)-2))  #日收益率
df2 = np.zeros((len(df),len(df.T)-2))  #日对数收益率
for i in range(len(df)):
    for j in range(len(df.T)-2):
        df1[i][j] = (df.loc[i][j+2]-df.loc[i][j+1])/(df.loc[i][j+1])
        df2[i][j] = math.log(1+df1[i][j])
df2
#%%
##根据算法对数据进行聚类
#归一化数据
n, m = len(df2), len(df2.T)
data1 = np.zeros((n,m))
for i in range(n):
    a = min(df2[i])
    b = max(df2[i])
    
    for j in range(m):
        data1[i][j]=(df2[i][j]-a)/(b-a)
data2 = pd.DataFrame(data1)
data2.to_csv(r"上海预处理后的数据归一化数据.csv")
#%%
def GYjuli(i,j):
    theta = 0.5
    m = len(df2.T)
    n = len(df2)
    a, b, a1, b1 = [], [], [], []
    X1 = np.zeros((1,m))
    X2 = np.zeros((1,m))
    A, B, A1, B1, D = 0, 0, 0, 0, 0
    # 取最大最小值
    for l in range(n):
        a.append(min(abs(data2.loc[i]-data2.loc[l])))
        b.append(max(abs(data2.loc[i]-data2.loc[l])))
        a1.append(min(abs(data2.loc[j]-data2.loc[l])))
        b1.append(max(abs(data2.loc[j]-data2.loc[l])))
    
    A, B = min(a), max(b)
    A1, B1 = min(a1), max(b1)
    c = abs(data2.loc[i]-data2.loc[j])
    
    for k in range(m):
        X1[0][k] = (A+theta*B)/(c[k]+theta*B) # 计算不同属性下的灰色关联系数
        X2[0][k] = (A1+theta*B1)/(c[k]+theta*B1)
    # print(X1)
    # print(X2)
    # D = np.sqrt(np.sum(np.square(X1 - X2))) #欧式距离
    D = np.sum(abs(X1 - X2)) #汉明距离
    return D
#%%
D = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        if i<j: 
            D[i][j] = GYjuli(i,j)
        if i>j:
            D[i][j] = D[j][i]
D1 = pd.DataFrame(D)
D1.to_csv(r"上海股票间的距离.csv")
#%%
def xiaoyong(x,singam,theta):
    lamda = np.zeros((len(x),6));
    lamda1 = np.ones((len(x),6));
    lamda2 = np.zeros((len(x),6));
    juhe = np.zeros((1,6));
    
    for i in range(0,len(x)): # 计算不同属性下的函数
        # 计算相对损失函数
        lamda[i,1] = singam * x[i];
        lamda[i,2] = x[i];
        lamda[i,3] = 1-x[i];
        lamda[i,4] = singam * (1-x[i]);
    
        # 计算基于后悔理论的效用函数
        lamda1[i,1] = 1 - (1 - np.exp(theta * lamda[i,1]))/theta
        lamda1[i,2] = 1 - (1 - np.exp(theta * lamda[i,2]))/theta
        lamda1[i,3] = 1 - (1 - np.exp(theta * lamda[i,3]))/theta
        lamda1[i,4] = 1 - (1 - np.exp(theta * lamda[i,4]))/theta
        
        # 计算相对效用函数
        lamda2[i,0] = 1 - lamda1[i,2] 
        lamda2[i,1] = lamda1[i,1] - lamda1[i,2] 
        lamda2[i,4] = lamda1[i,4] - lamda1[i,3] 
        lamda2[i,5] = 1 - lamda1[i,3]
        
    # 计算聚合相对效用函数(均值聚合)
    juhe = np.mean(lamda2,axis=0)
    
    return juhe
#%%
# 计算不同对象的聚合相对效用函数矩阵
singam = 0.15 
theta = 0.5
lamda = np.zeros((n,6));
for i in range(0,n):
    x = data2.iloc[i]
    lamda[i] = xiaoyong(x ,singam, theta)

#根据聚合相对效用函数计算不同对象的边界阈值
yuzhi = np.zeros((n,2))
for i in range(0,n):
    for j in range(0,len(lamda.T)):
        yuzhi[i][0] = (lamda[i][4])/(lamda[i][4]+lamda[i][0]-lamda[i][1])
        yuzhi[i][1] = (lamda[i][5]-lamda[i][4])/(lamda[i][1]+lamda[i][5]-lamda[i][4])
yuzhi = pd.DataFrame(yuzhi)
yuzhi.to_csv(r"上海股票的边界阈值.csv")
#%%
##根据归一化数据计算每个对象到正理想解和负理想解的距离
d1 = np.zeros((n,1)) 
d2 = np.zeros((n,1)) 
lishu = np.zeros((n,1))

for i in range(0,n):
    a = 0
    b = 0
    for j in range(0,m):
        a = a + (1 - data2.T[i][j])*(1 - data2.T[i][j])
        b = b + data2.T[i][j] * data2.T[i][j]
    d1[i] = np.sqrt(a) #每个对象到正理想解的距离
    d2[i] = np.sqrt(b) #每个对象到负理想解的距离
    lishu[i] = d1[i]/(d1[i]+d2[i])
lishuu = pd.DataFrame(lishu)
lishuu.to_csv(r"上海股票的隶属函数.csv")  
#%%
def qitayuzhi1(yz):
    #根据阈值设置下一个阈值的范围
    if yz[1] < 0.5:
        yuzhi2 = np.arange(yz[1],0.5,0.05) 
        V = []
        for k in range(0,len(yuzhi2)):
            yuzhi1 = 1-yuzhi2[k]
            x = 0 
            v1 = v2 = []
            v3 = 0
            
            #计算升高区域、降低区域、阴影区域的面积
            for i in range(0,n):
                if lishu[i] >= yuzhi1:
                    v1.append(1-lishu[i])
                elif lishu[i] <= yuzhi2[k]:
                    v2.append(lishu[i])
                else:
                    v3 = v3 + 1
            V1 = np.sum(v1)
            V2 = np.sum(v2)
            V3 = v3
    
            V.append(abs(V1+V2-V3))

    else:
        yuzhi2 = np.arange(yz[1],yz[0],0.05)
        x = 1
        V = []
        l = []
        
        for k in range(0,len(yuzhi2)):   
            yuzhi1 = np.arange(yuzhi2[k]+0.05,yz[0]+0.05,0.05)
            l.append(len(yuzhi1))
            for j in range(0,len(yuzhi1)):
                v1 = v2 = []
                v3 = 0
                
                #计算升高区域、降低区域、阴影区域的面积
                for i in range(0,n):
                    if lishu[i] >= yuzhi1[j]:
                        v1.append(1-lishu[i])
                    elif lishu[i] <= yuzhi2[k]:
                        v2.append(lishu[i])
                    else:
                        v3 = v3 + 1
                V1 = np.sum(v1)
                V2 = np.sum(v2)
                V3 = v3
    
                V.append(abs(V1+V2-V3))
        
    #根据最优化目标函数得到阈值
    if x == 0: 
        yz2 = yuzhi2[V.index(min(V))]
        yz1 = 1-yz2
    else:
        xx = V.index(min(V))
        for i in range(0,len(l)):
            if xx-l[i]>=0:
                xx = xx-l[i]
            else: 
                yz2 = yuzhi2[i]
                yuzhi1 = np.arange(yuzhi2[i]+0.05,yz[0]+0.05,0.05)
                yz1 = yuzhi1[xx]
    return yz1,yz2
#%%
df1 = pd.read_csv("上海股票的边界阈值.csv",index_col = 0)
df1.head()
#%%
dis = pd.read_csv("上海股票间的距离.csv")
dis = dis.drop(['Unnamed: 0'],axis=1)
dis.head()
#%%
y = 0
for l in range(0,n):
    sc = [];uc = []; dc =[]; 
    # 根据边界阈值进行三支决策
    for i in range(n):
        if dis.loc[l][i]>=df1.loc[l][0]:
            dc.append(i)
        elif dis.loc[l][i]<df1.loc[l][0] and dis.loc[l][i]>df1.loc[l][1]:
            uc.append(i)
        else:
            sc.append(i) 
        
    # 根据其他阈值继续进行三支决策
    yz = qitayuzhi1(df1.loc[l])
    #print(yz)
    while y == 0:
        yuzhi = yz
        for i in range(n):
            if i in uc:
                if dis.loc[l][i] <= yuzhi[1]:
                    sc.append(i)
                    uc.remove(i)
                if dis.loc[l][i] >= yuzhi[0]: 
                    dc.append(i)
                    uc.remove(i)
        if yuzhi[0]- 0.5 > 0.1 or len(uc) >= 1/4*n:
            yz = qitayuzhi1(yuzhi)
            # print(yz)
        else:
            y = 1
    print(l,'............')        
    print(sc)
    print(uc)
    print(dc)
#%%
