#%%
## 导入需要的库
import numpy as np
import pandas as pd
import math
import os
import random
from matplotlib.pyplot import *
import matplotlib.pyplot as plt

#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
rc('font',size=16); rc('text', usetex=True)  #调用tex字库
#%%
data = pd.read_csv("随机挑选后的股票数据.csv")
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
#%% md
# ## 不同模型的聚类结果
#%%
# 我们模型的结果
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

# 逆方差聚类结果
E1 = [1, 3, 4, 5, 6, 9, 11, 12, 13, 16, 17, 21, 22, 26, 27, 29, 33, 35, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49]
E2 = [0, 2, 7, 8, 10, 14, 15, 18, 19, 20, 23, 24, 25, 28, 30, 31, 32, 34, 36, 37, 38, 41]
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
ExpCovar.to_csv(r"50支股票的协方差矩阵.csv")
#%% md
# ## K-means聚类结果下的投资组合
#%%
w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
y = 0
while y == 0: 
    for k in range(0,len(A1)):
        a = A1[k]
        for j in range(0,len(A2)):
            b = A2[j]
            for i in range(0,len(w)):
                Y = w[i]*data1.iloc[a]+(1-w[i])*data1.iloc[b]
                s = pd.Series(Y)
                if np.mean(Y) > 1.8 and (np.std(Y))**2 < 0.25:
                        print(a, b)
                        print(w[i])
                        print([np.mean(Y), np.std(Y), np.max(Y), np.min(Y), s.skew(), s.kurt()])
                        print((np.mean(Y)-0.73)/np.std(Y))
                        print(np.std(Y[1135:]))
                        print('*****************************')
                        y = 1
                else:
                    break
    y = 1
#%%
w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
y = 0
while y == 0: 
    for k in range(0,len(A3)):
        a = A3[k]
        for j in range(0,len(A2)):
            b = A2[j]
            for i in range(0,len(w)):
                Y = w[i]*data1.iloc[a]+(1-w[i])*data1.iloc[b]
                s = pd.Series(Y)
                if np.mean(Y) > 1.8 and (np.std(Y))**2 < 0.25:
                        print(a, b)
                        print(w[i])
                        print([np.mean(Y), np.std(Y), np.max(Y), np.min(Y), s.skew(), s.kurt()])
                        print((np.mean(Y)-0.73)/np.std(Y))
                        print(np.std(Y[1135:]))
                        print('*****************************')
                        y = 1
                else:
                    break
    y = 1
#%% md
# ## 三支聚类结果下的投资组合
#%%
w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
y = 0
while y == 0: 
    for k in range(0,len(B1)):
        a = B1[k]
        for j in range(0,len(B2)):
            b = B2[j]
            for i in range(0,len(w)):
                Y = w[i]*data1.iloc[a]+(1-w[i])*data1.iloc[b]
                s = pd.Series(Y)
                if np.mean(Y) > 1.7 and (np.std(Y))**2 < 0.25:
                        print(a, b)
                        print(w[i])
                        print([np.mean(Y), np.std(Y), np.max(Y), np.min(Y), s.skew(), s.kurt()])
                        print((np.mean(Y)-0.73)/np.std(Y))
                        print(np.std(Y[1135:]))
                        print('*****************************')
                        y = 1
                else:
                    break
    y = 1
#%% md
# ## 基于遗憾理论的灰度关联聚类结果的投资组合
#%%
w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
y = 0
while y == 0: 
    for k in range(0,len(C1)):
        a = C1[k]
        for j in range(0,len(C2)):
            b = C2[j]
            for i in range(0,len(w)):
                Y = w[i]*data1.iloc[a]+(1-w[i])*data1.iloc[b]
                s = pd.Series(Y)
                if np.mean(Y) > 1.8 and (np.std(Y))**2 < 0.25:
                        print(a, b)
                        print(w[i])
                        print([np.mean(Y), np.std(Y), np.max(Y), np.min(Y), s.skew(), s.kurt()])
                        print((np.mean(Y)-0.73)/np.std(Y))
                        print(np.std(Y[1135:]))
                        print('*****************************')
                        y = 1
                else:
                    break
    y = 1
#%% md
# ## 基于亲和传播聚类结果的投资组合
#%%
w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
y = 0
while y == 0: 
    for k in range(0,len(D1)):
        a = D1[k]
        for j in range(0,len(D2)):
            b = D2[j]
            for i in range(0,len(w)):
                Y = w[i]*data1.iloc[a]+(1-w[i])*data1.iloc[b]
                s = pd.Series(Y)
                if np.mean(Y) > 1.5 and (np.std(Y))**2 <= 0.25:
                        print(a, b)
                        print(w[i])
                        print([np.mean(Y), np.std(Y), np.max(Y), np.min(Y), s.skew(), s.kurt()])
                        print((np.mean(Y)-0.73)/np.std(Y))
                        print(np.std(Y[1135:]))
                        print('*****************************')
                        y = 1
                else:
                    break
    y = 1
#%% md
# ## 基于逆协方差聚类结果的投资组合
#%%
w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
y = 0
while y == 0: 
    for k in range(0,len(E1)):
        a = E1[k]
        for j in range(0,len(E2)):
            b = E2[j]
            for i in range(0,len(w)):
                Y = w[i]*data1.iloc[a]+(1-w[i])*data1.iloc[b]
                s = pd.Series(Y)
                if np.mean(Y) > 1.8 and (np.std(Y))**2 < 0.25:
                        print(a, b)
                        print(w[i])
                        print([np.mean(Y), np.std(Y), np.max(Y), np.min(Y), s.skew(), s.kurt()])
                        print((np.mean(Y)-0.73)/np.std(Y))
                        print(np.std(Y[1135:]))
                        print('*****************************')
                        y = 1
                else:
                    break
    y = 1
#%% md
# ## 不同投资组合的比较分析
#%%
# K-means结果下的投资组合
plt.rcParams['figure.figsize'] = (8.0, 4.0)
Y1 = 0.2 * data1.iloc[21] + 0.8 * data1.iloc[0]
s = pd.Series(Y1)
print([np.mean(Y1), np.std(Y1), np.max(Y1), np.min(Y1), s.skew(), s.kurt()])
print((np.mean(Y1)-0.73)/np.std(Y1))
print(np.std(Y1[1135:]))

x = np.linspace(0,1500,1500)
plot(x,Y1,'k')
show()
#%%
# 三支聚类结果下的投资组合
Y2 = 0.3 * data1.iloc[48] + 0.7 * data1.iloc[27]
s = pd.Series(Y2)
print([np.mean(Y2), np.std(Y2), np.max(Y2), np.min(Y2), s.skew(), s.kurt()])
print((np.mean(Y2)-0.73)/np.std(Y2))
print(np.std(Y2[1135:]))

plot(x,Y2,'k');
show()
#%%
# 基于遗憾理论的灰度关联聚类结果的投资组合
Y3 = 0.3 * data1.iloc[3] + 0.7 * data1.iloc[0]
s = pd.Series(Y3)
print([np.mean(Y3), np.std(Y3), np.max(Y3), np.min(Y3), s.skew(), s.kurt()])
print((np.mean(Y3)-0.73)/np.std(Y3))
print(np.std(Y3[1135:]))

plot(x,Y3,'k');
show()
#%%
# 基于亲和传播聚类结果的投资组合
Y4 = 0.2 * data1.iloc[5] + 0.8 * data1.iloc[26]
s = pd.Series(Y4)
print([np.mean(Y4), np.std(Y4), np.max(Y4), np.min(Y4), s.skew(), s.kurt()])
print((np.mean(Y4)-0.73)/np.std(Y4))
print(np.std(Y4[1135:]))

plot(x,Y4,'k');
show()
#%%
# 基于逆协方差聚类结果的投资组合
Y5 = 0.2 * data1.iloc[16] + 0.8 * data1.iloc[0]
s = pd.Series(Y5)
print([np.mean(Y5), np.std(Y5), np.max(Y5), np.min(Y5), s.skew(), s.kurt()])
print((np.mean(Y5)-0.73)/np.std(Y5))
print(np.std(Y5[1135:]))

plot(x,Y5,'k');
show()
#%%

#%%
Y = 0.9*data1.iloc[0]+0.1*data1.iloc[49]
Y1 = 0.2 * data1.iloc[21] + 0.8 * data1.iloc[0]
Y2 = 0.3 * data1.iloc[48] + 0.7 * data1.iloc[27]
Y3 = 0.3 * data1.iloc[3] + 0.7 * data1.iloc[0]
Y4 = 0.2 * data1.iloc[5] + 0.8 * data1.iloc[26]
Y5 = 0.2 * data1.iloc[16] + 0.8 * data1.iloc[0]
#%%
x = np.linspace(0,1500,1500)
plt.rcParams['figure.figsize'] = (20.0, 10.0)
#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
rc('font',size=16); rc('text', usetex=True)  #调用tex字库

# 按两行三咧显示
a = subplot(2,3,1) #在第一窗口显示
a.plot(x,Y,'r', label = '$P_1$')
plt.legend()
b = subplot(2,3,2) #在第二个窗口显示
b.plot(x,Y1,'y',label = '$P_2$' )
plt.legend()
c = subplot(2,3,3)
c.plot(x,Y2,'k', label = '$P_3$')
plt.legend()
d = subplot(2,3,4) 
d.plot(x,Y3,'c', label = '$P_4$')
plt.legend()
e = subplot(2,3,5)
e.plot(x,Y4,'g', label = '$P_5$')
plt.legend()
# f = subplot(2,3,6) 
# f.plot(x,Y5,'b', label = '$P_6$')
plt.legend()
savefig("投资组合比较1.jpg",dpi=300, bbox_inches='tight')
show()
#%%
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
gs = gridspec.GridSpec(2, 6) # 创立2 * 6 网格
gs.update(wspace=0.8)
# 对第一行进行绘制
ax1 = plt.subplot(gs[0, :2]) # gs(哪一行，绘制网格列的范围)
ax1.plot(x,Y,'r', label = '$P_1$')
plt.legend()
ax2 = plt.subplot(gs[0, 2:4])
ax2.plot(x,Y1,'y',label = '$P_2$' )
plt.legend()
ax3 = plt.subplot(gs[0, 4:6])
ax3.plot(x,Y2,'k', label = '$P_3$')
plt.legend()
# 对第二行进行绘制
ax4 = plt.subplot(gs[1, 1:3])
ax4.plot(x,Y3,'c', label = '$P_4$')
plt.legend()
ax5 = plt.subplot(gs[1, 3:5])
ax5.plot(x,Y4,'g', label = '$P_5$')
plt.legend()
savefig("投资组合比较1.jpg",dpi=300, bbox_inches='tight')
plt.show()
#%%

#%%

#%%

#%%
s1, s2, s3, s4, s5 = pd.Series(Y1), pd.Series(Y2), pd.Series(Y3), pd.Series(Y4),pd.Series(Y5)
L1 = [np.mean(Y1), (np.std(Y1))**2, np.std(Y1), np.max(Y1), np.min(Y1), s1.skew(), s1.kurt(),    
     (np.mean(Y1)-0.73)/np.std(Y1), np.std(Y1[1135:])]

L2 = [np.mean(Y2), (np.std(Y2))**2, np.std(Y2), np.max(Y2), np.min(Y2), s2.skew(), s2.kurt(),    
     (np.mean(Y2)-0.73)/np.std(Y2), np.std(Y2[1135:])]

L3 = [np.mean(Y3), (np.std(Y3))**2, np.std(Y3), np.max(Y3), np.min(Y3), s3.skew(), s3.kurt(),    
     (np.mean(Y3)-0.73)/np.std(Y3), np.std(Y3[1135:])]

L4 = [np.mean(Y4), (np.std(Y4))**2, np.std(Y4), np.max(Y4), np.min(Y4), s4.skew(), s4.kurt(),    
     (np.mean(Y4)-0.73)/np.std(Y4), np.std(Y4[1135:])]

L5 = [np.mean(Y5), (np.std(Y5))**2, np.std(Y5), np.max(Y5), np.min(Y5), s5.skew(), s5.kurt(),    
     (np.mean(Y5)-0.73)/np.std(Y5), np.std(Y5[1135:])]

for i in range(len(L1)):
    L1[i] = round(L1[i],4)
    L2[i] = round(L2[i],4)
    L3[i] = round(L3[i],4)
    L4[i] = round(L4[i],4)
    L5[i] = round(L5[i],4)
print(L1)
print(L2)
print(L3)
print(L4)
print(L5)
#%%

#%%
