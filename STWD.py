#%%
## 导入需要的库
import numpy as np
import pandas as pd
import math
import os
import random
#%%
## 读取对数收益率数据
data = pd.read_excel(r"D:\数据\数据\随机挑选后的股票数据（消融实验）.xlsx")
data.head()
#%% md
# ### 归一化数据
#%%
N = len(data);M = len(data.T);
data1 = np.zeros((N,M))
for i in range(N):
    a = min(data.iloc[i])
    b = max(data.iloc[i])
    for j in range(M):
        data1[i][j]=(data[j+1][i]-a)/(b-a)
data2 = pd.DataFrame(data1)
data2.to_csv(r"D:\数据\数据\数据与代码\归一化.csv")
#%% md
# ### 计算条件概率
#%%
###根据归一化后的数据得到决策信息矩阵
G = data1
### 设定正理想解和负理想解
X1 = np.ones((1,M))
X0 = np.zeros((1,M))

### 计算决策对象到正负理想解的灰色距离关联度
D1 = np.zeros((1,N));D0 = np.zeros((1,N));  r1 = np.zeros((1,N)); r0 = np.zeros((1,N)); Pr = np.zeros((1,N))
for i in range(0,N):
    D1[0][i] = np.sum(X1-G[i])
    D0[0][i] = np.sum(G[i])
for i in range(0,N):
    r1[0][i] = (D1.min()+D1.max())/(D1[0][i]+D1.min()+D1.max())
    r0[0][i] = (D0.min()+D0.max())/(D0[0][i]+D0.min()+D0.max())
    Pr[0][i] = r0[0][i]/( r0[0][i]+ r1[0][i])
print(Pr)
#%%
### 计算决策对象到正负理想解的灰色距离关联度
D1 = np.zeros((1,N));D0 = np.zeros((1,N));  r1 = np.zeros((1,N)); r0 = np.zeros((1,N)); Pr = np.zeros((1,N))
for i in range(0,N):
    D1[0][i] = np.sum(X1-G[i])
    D0[0][i] = np.sum(G[i])
for i in range(0,N):
    r1[0][i] = (D1.min()+D1.max())/(D1[0][i]+D1.min()+D1.max())
    r0[0][i] = (D0.min()+D0.max())/(D0[0][i]+D0.min()+D0.max())
    Pr[0][i] = r0[0][i]/( r0[0][i]+ r1[0][i])
    
l = np.zeros((N,1))
for i in range(0,N):
    a = 0
    for j in range(0,M):
        if G[i][j]>0.5:
            a = a +1 
    l[i] = a

for i in range(0,N):
    if l[i]>=900:
        Pr[0][i] = (Pr[0][i]+1)/2
print(Pr)
#%% md
# ### 计算阈值对
#%%
## 利用CRITIC权重法计算属性权重
n = len(data);m = len(data.T);

A = np.zeros((m,1))
S = np.zeros((m,1))
R = np.corrcoef(data2.T)  #计算属性指标之间的相关系数得到相关性矩阵
a1 = np.mean(data2,axis=0) 

for j in range(0,m):
    A[j] = m-np.sum(R[j])  #计算冲突性
    b1 = []
    for i in range(0,n):
        b1.append((data2.T[i][j]-a1[j])**2)
    S[j] = np.sqrt(np.sum(b1)/(n-1))

    W = np.zeros((m,1))
for j in range(0,m):
    W[j] = (S[j]*A[j])/np.sum(S*A)
    
W1 = pd.DataFrame(W)
W1.to_csv(r"D:\数据\数据\数据与代码\权重.csv")
#%%
##计算基于正负理想解的灰色关联度
a = []; b = []; c1 = []; c2 = []; theta = 0.5
for i in range(0,n): 
    X = []
    # 除去0,1之外取最值
    a1 = abs(data2.loc[i]-1)
    b1 = data2.loc[i]
    a2 = 1
    b2 = 0
    for k in range(0,m):
        if a1[k] < a2 and a1[k] != 0:
            a2 = a1[k]
        if b1[k] > b2 and b1[k] != 1:
            b2 = b1[k]   
    a.append(a2)
    b.append(b2)
    c1 = abs(data2.loc[i]-1)
    c2 = data2.loc[i]
#%%
D1 = []; D2 = []
D11 = []; D22 = []
for i in range(0,n):
    d1 = []; d2 = [];
    for j in range(0,m):
        d1.append((a[i]+theta*b[i])/(c1[j]+theta*b[i]))  #计算正理想解下的灰色关联系数
        d2.append((a[i]+theta*b[i])/(c2[j]+theta*b[i]))  #计算负理想解下的灰色关联系数
    D11.append(np.max(d1))
    D22.append(np.min(d2))
    
    D1.append(np.sum(d1*W.T))
    D2.append(np.sum(d2*W.T))

#计算决策对象的隶属度
lishu = np.zeros((n,1))
lishu1 = np.zeros((n,1))
for i in range(0,n):
    lishu[i] = D11[i]/(D11[i]+D22[i])
    lishu1[i] = D1[i]/(D1[i]+D2[i])
print(max(lishu))
print(min(lishu))

print(max(lishu1))
print(min(lishu1))   #根据属性权重计算的隶属函数

lishuu = pd.DataFrame(lishu)
lishuu.to_csv(r"D:\数据\数据\数据与代码\隶属函数.csv")

lishuu1 = pd.DataFrame(lishu1)
lishuu1.to_csv(r"D:\数据\数据\数据与代码\基于权重的隶属函数.csv")
#%%
def sunshi(x,singam,theta):
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
n = len(data) #股票个数 
m = len(data.T) #股票指标个数
singam = [0.15]*m
theta = 0.5
lamda = np.zeros((n,6));
for i in range(0,n):
    x = data2.iloc[i]
    lamda[i] = sunshi(x ,singam, theta)

#根据聚合相对损失函数计算不同对象的初始阈值
yuzhi = np.zeros((n,2))
for i in range(0,n):
    for j in range(0,len(lamda.T)):
        yuzhi[i][0] = (lamda[i][3]-lamda[i][4])/(lamda[i][3]-lamda[i][4]+lamda[i][1])
        yuzhi[i][1] = (lamda[i][4])/(lamda[i][4]+lamda[i][2]-lamda[i][1])
yuzhi = pd.DataFrame(yuzhi)
yuzhi.to_csv(r"D:\数据\数据\数据与代码\不同对象的初始阈值.csv")
#%%
def qitayuzhi(yz):
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
                if lishu1[i] >= yuzhi1:
                    v1.append(1-lishu1[i])
                elif lishu1[i] <= yuzhi2[k]:
                    v2.append(lishu1[i])
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
                    if lishu1[i] >= yuzhi1[j]:
                        v1.append(1-lishu1[i])
                    elif lishu1[i] <= yuzhi2[k]:
                        v2.append(lishu1[i])
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
# ### 进行聚类
#%%
df1 = pd.read_csv(r"D:\数据\数据\数据与代码\不同对象的初始阈值.csv",index_col = 0)
df1.head()
#%%
y = 0
l = random.randint(0,50)

sc = [];uc = []; dc =[]; 
# 根据初始阈值进行三支决策
for i in range(n):
    if Pr[0][i]>=df1.loc[l][0]:
        sc.append(i)
    elif Pr[0][i]<df1.loc[l][0] and Pr[0][i]>df1.loc[l][1]:
        uc.append(i)
    else:
        dc.append(i) 

print(df1.loc[l])
# 根据其他阈值继续进行三支决策
yz = qitayuzhi(df1.loc[l])
print(yz)
while y == 0:
    yuzhi = yz
    for i in range(n):
        if i in uc:
            if Pr[0][i] >= yuzhi[0]:
                sc.append(i)
                uc.remove(i)
            if Pr[0][i] <= yuzhi[1]: 
                dc.append(i)
                uc.remove(i)
    if len(uc) >= 1/4*n:
        yz = qitayuzhi(yuzhi)
        print(yz)
    else:
        y = 1
print(sc)
print(uc)
print(dc)
#%%

#%%

#%%

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
y = 0
l = random.randint(0,50)

sc = [];uc = []; dc =[]; 
# 根据初始阈值进行三支决策
for i in range(n):
    if Pr[0][i]>=df1.loc[l][0]:
        sc.append(i)
    elif Pr[0][i]<df1.loc[l][0] and Pr[0][i]>df1.loc[l][1]:
        uc.append(i)
    else:
        dc.append(i) 
        
print(sc)
print(uc)
print(dc)

print(df1.loc[l])
#%%
# 根据其他阈值继续进行三支决策
yuzhi = qitayuzhi1(df1.loc[l])
print(yuzhi)
#%%
for i in range(n):
    if i in uc:
        if Pr[0][i] >= yuzhi[0]:
            sc.append(i)
            uc.remove(i)
        if Pr[0][i] <= yuzhi[1]: 
            dc.append(i)
            uc.remove(i)
print(sc)
print(uc)
print(dc)
#%%

#%%


while y == 0:
    for i in range(n):
        if i in uc:
            if Pr[0][i] >= yuzhi[0]:
                sc.append(i)
                uc.remove(i)
            if Pr[0][i] <= yuzhi[1]: 
                dc.append(i)
                uc.remove(i)
    if len(uc) >= 1/4*n:
        yuzhi = qitayuzhi1(yuzhi)
        print(yuzhi)
    else:
        y = 1
print(sc)
print(uc)
print(dc)
#%%
