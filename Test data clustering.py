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
## 随机挑选50支股票
import random
lis = list(range(0,400))
sample_num = 50
print(random.sample(lis, sample_num))
#%%
## 随机挑选的50支股票信息如下
gupiao = [46, 130, 201, 267, 15, 90, 242, 235, 42, 237, 221, 285, 261, 169, 194, 314, 129, 97, 204, 125,    
          119, 383, 5, 397, 271, 216, 101, 334, 376, 208, 188, 347, 28, 206, 385, 391, 395, 152, 137,    
          275, 338, 80, 89, 104, 92, 312, 363, 72, 279, 87]

## 读取所有股票的对数收益率数据
df = pd.read_excel("对数收益率（1500个交易日）.xlsx")
df.head()
#%%
GPdaima = []
for i in range(0,n):
    GPdaima.append(df.loc[gupiao1[i]]['股票代码'])
print(GPdaima)
#%%
## 根据随机挑选的股票得到相应数据
gupiao1 = sorted(gupiao) #排序
n, m = len(gupiao), len(df.T)
data = np.zeros((n,m))
for i in range(0,n):
    for j in range(0,m):
        data[i][j] = df.loc[gupiao1[i]][j]

# 将矩阵读入csv中        
data = pd.DataFrame(data)
data.to_csv(r"随机挑选后的股票数据.csv")  
#%% md
# ## 归一化数据
#%%
n, m = len(data), len(data.T)
data1 = np.zeros((n,m))
for i in range(n):
    a = min(data.T[i])
    b = max(data.T[i])
    
    for j in range(m):
        data1[i][j]=(data[j][i]-a)/(b-a)
data2 = pd.DataFrame(data1)
data2.to_csv(r"归一化数据.csv")
#%%
data2
#%%
def GYjuli(i,j):
    theta = 0.5
    m = len(data.T)
    n = len(data)
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
D1.to_csv(r"50支股票间的距离.csv")
#%% md
# ## 计算聚合相对效用函数
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
#%% md
# ## 计算边界阈值
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
yuzhi.to_csv(r"50支股票的边界阈值.csv")
#%% md
# ## 基于阴影集计算其他阈值
#%% md
# ### 计算隶属度
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
#%%
lishuu = pd.DataFrame(lishu)
lishuu.to_csv(r"50支股票的隶属函数.csv")    
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
#%% md
# ## 进行聚类
#%%
df1 = pd.read_csv("50支股票的边界阈值.csv",index_col = 0)
df1.head()
#%%
dis = pd.read_csv("50支股票间的距离.csv")
dis = dis.drop(['Unnamed: 0'],axis=1)
dis.head()
#%%

#%%
y = 0
l = 20
    
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
print(sc)
print(uc)
print(dc)
#%%
print(len(sc))
print(len(uc))
print(len(dc))
#%%
n1 = len(uc)+len(dc)
U = uc + dc  #重新设定论域
df2 = np.zeros((n1,2))
for i in range(0,n1):
    df2[i][0] = df1.loc[U[i]][0]
    df2[i][1] = df1.loc[U[i]][1]
#%%
dis1 = np.zeros((n1,n1))
for i in range(0,n1):
    for j in range(0,n1):
        dis1[i][j] = dis.loc[U[i]][U[j]]
dis1
#%%
y = 0
l = 15
# print(l)
    
sc1 = [];uc1 = []; dc1 =[]; 
# 根据边界阈值进行三支决策
for i in range(n1):
    if dis1[l][i]>=df2[l][0]:
        dc1.append(i)
    elif dis1[l][i]<df2[l][0] and dis1[l][i]>df2[l][1]:
        uc1.append(i)
    else:
        sc1.append(i) 
print(sc1)
print(uc1)
print(dc1)
#%%
# 根据其他阈值继续进行三支决策
yz = qitayuzhi1(df2[l])
#print(yz)
while y == 0:
    yuzhi = yz
    for i in range(n1):
        if i in uc1:
            if dis1[l][i] <= yuzhi[1]:
                sc1.append(i)
                uc1.remove(i)
            if dis1[l][i] >= yuzhi[0]: 
                dc1.append(i)
                uc1.remove(i)
    if yuzhi[0]- 0.5 > 0.1 or len(uc1) >= 1/4*n:
        yz = qitayuzhi1(yuzhi)
        # print(yz)
    else:
        y = 1
print(sc1)
print(uc1)
print(dc1)
#%%
def GYjuli1(i,j):
    theta = 0.5
    m = len(data3.T)
    n = len(data3)
    a, b, a1, b1 = [], [], [], []
    X1 = np.zeros((1,m))
    X2 = np.zeros((1,m))
    A, B, A1, B1, D = 0, 0, 0, 0, 0
    # 取最大最小值
    for l in range(n):
        a.append(min(abs(data3.loc[i]-data3.loc[l])))
        b.append(max(abs(data3.loc[i]-data3.loc[l])))
        a1.append(min(abs(data3.loc[j]-data3.loc[l])))
        b1.append(max(abs(data3.loc[j]-data3.loc[l])))
    
    A, B = min(a), max(b)
    A1, B1 = min(a1), max(b1)
    c = abs(data3.loc[i]-data3.loc[j])
    
    for k in range(m):
        X1[0][k] = (A+theta*B)/(c[k]+theta*B) # 计算不同属性下的灰色关联系数
        X2[0][k] = (A1+theta*B1)/(c[k]+theta*B1)
    # print(X1)
    # print(X2)
    # D = np.sqrt(np.sum(np.square(X1 - X2))) #欧式距离
    D = np.sum(abs(X1 - X2)) #汉明距离
    return D
#%%
data3 = np.zeros((n1,m))
for i in range(0,n1):
    for j in range(0,m):
        data3[i][j] = data2.loc[U[i]][j]
data3 = pd.DataFrame(data3)
data3
#%%
D2 = np.zeros((n1,n1))
for i in range(n1):
    for j in range(n1):
        if i<j: 
            D2[i][j] = GYjuli1(i,j)
        if i>j:
            D2[i][j] = D2[j][i]
D2 = pd.DataFrame(D2)
D2
#%%

#%%
y = 0
l = 15
    
sc1 = [];uc1 = []; dc1 =[]; 
# 根据边界阈值进行三支决策
for i in range(n1):
    if D2.loc[l][i]>=df2[l][0]:
        dc1.append(i)
    elif D2.loc[l][i]<df2[l][0] and D2.loc[l][i]>df2[l][1]:
        uc1.append(i)
    else:
        sc1.append(i) 
print(sc1)
print(uc1)
print(dc1)
#%%
# 根据其他阈值继续进行三支决策
yz = qitayuzhi1(df2[l])
print(yz)
yuzhi = yz
for i in range(n1):
    if i in uc1:
        if D2.loc[l][i] <= yuzhi[1]:
            sc1.append(i)
            uc1.remove(i)
        if D2.loc[l][i] >= yuzhi[0]: 
            dc1.append(i)
            uc1.remove(i)
print(sc1)
print(uc1)
print(dc1)
#%%
for i in range(0,n1):
    for j in range(0,n1):
        if D2.loc[i][j] < 0.52 and D2[i][j] >0.47:
            print('*')
#%%
label2 = [0, 1, 2, 4, 5, 7, 8, 15, 16, 17, 18, 20, 21, 22, 24, 26, 32, 33]
label3 = [3, 6, 9, 10, 11, 12, 13, 14, 19, 23, 25, 27, 28, 29, 30, 31]
label22, label33 = [], [] 

for i in range(n1):
    if i in label2:
        label22.append(U[i])
    else:
        label33.append(U[i])
print(label22)
print(label33)
#%%
## 聚类结果如下所示
label11 = [20, 1, 4, 5, 6, 9, 16, 22, 25, 34, 36, 38, 43, 44, 46, 49]
label22 = [12, 17, 27, 2, 3, 8, 10, 21, 23, 24, 26, 29, 30, 31, 33, 37, 47, 48]
label33 = [0, 7, 11, 13, 14, 15, 18, 19, 28, 32, 35, 39, 40, 41, 42, 45]
#%%
lb1, lb2, lb3 = [], [], []
for i in range(0,n):
    if i in label11:
        lb1.append(GPdaima[i])
    if i in label22:
        lb2.append(GPdaima[i])
    if i in label33:
        lb3.append(GPdaima[i])
print(lb1)
print(lb2)
print(lb3)
#%%
print(len(lb1))
print(len(lb2))
print(len(lb3))
#%%

#%%

#%%
# 根据其他阈值继续进行三支决策
yuzhi = [0.65, 0.35]
for i in range(n1):
    if i in uc1:
        if D2.loc[l][i] <= yuzhi[1]:
            sc1.append(i)
            uc1.remove(i)
        if D2.loc[l][i] >= yuzhi[0]: 
            dc1.append(i)
            uc1.remove(i)
print(sc1)
print(uc1)
print(dc1)
#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%
### 策略1下的股票净值图
X = [8, 18, 20, 24, 29]

for i in range(len(X)):
    Y = []
    a = 1
    for j in range(0,len(data2.T)):
        b = X[i]
        a = a + data.loc[b][j+1]
        Y.append(a)
    
    x = np.linspace(0,1500,1500)
    rc('font',size=16); rc('text', usetex=True)  #调用tex字库
    plot(x,Y,'k');
    show()
#%%
