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
## 读取对数收益率数据
data = pd.read_csv("归一化数据.csv")
data = data.drop(['Unnamed: 0'], axis = 1)
data.head()
#%%
def xiaoyong(x,singam,theta):
    lamda = np.zeros((len(x),6));
    u = np.ones((len(x),6));
    u1 = np.zeros((len(x),6));
    
    for i in range(0,len(x)): # 计算不同属性下的函数
        # 计算相对损失函数
        lamda[i,1] = singam * x[i];
        lamda[i,2] = x[i];
        lamda[i,3] = 1-x[i];
        lamda[i,4] = singam * (1-x[i]);
    
        # 计算基于后悔理论的效用函数
        u[i,1] = 1 - (1 - np.exp(-theta * lamda[i,1]))/theta
        u[i,2] = 1 - (1 - np.exp(-theta * lamda[i,2]))/theta
        u[i,3] = 1 - (1 - np.exp(-theta * lamda[i,3]))/theta
        u[i,4] = 1 - (1 - np.exp(-theta * lamda[i,4]))/theta
        
        # 计算相对效用函数
        u1[i,0] = 1 - u[i,2] 
        u1[i,1] = u[i,1] - u[i,2] 
        u1[i,4] = u[i,4] - u[i,3] 
        u1[i,5] = 1 - u[i,3]
        
    return u1
#%%
# 计算不同对象的聚合相对效用函数矩阵
X = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
theta = 0.5
n, m = len(data), len(data.T)
lamda = np.zeros((len(X),6));
u = np.ones((len(X),6));
u1 = np.zeros((len(X),6));
u3 = np.zeros((len(X),6))
x = data.iloc[0]
yuzhi = np.zeros((len(X),2))

for k in range(0,len(X)):
    singam = X[k]
   
    # 计算相对损失函数
    lamda[k,1] = singam * x[0];
    lamda[k,2] = x[0];
    lamda[k,3] = 1-x[0];
    lamda[k,4] = singam * (1-x[0]);
    
    # 计算基于后悔理论的效用函数
    u[k,1] = 1 - (1 - np.exp(-theta * lamda[k,1]))/theta
    u[k,2] = 1 - (1 - np.exp(-theta * lamda[k,2]))/theta
    u[k,3] = 1 - (1 - np.exp(-theta * lamda[k,3]))/theta
    u[k,4] = 1 - (1 - np.exp(-theta * lamda[k,4]))/theta
        
    # 计算相对效用函数
    u1[k,0] = 1 - u[k,2] 
    u1[k,1] = u[k,1] - u[k,2] 
    u1[k,4] = u[k,4] - u[k,3] 
    u1[k,5] = 1 - u[k,3]
    u2 = np.max(u1, axis = 0)
    delta = 0.01
    for j in range(0,6):
        u3[k][j] = u1[k,j] + 1 - np.exp(- delta * (u1[k,j]-u2[j]))
            
    #根据基于后悔理论的相对效用函数计算不同对象的不同粒度下的阈值
    yuzhi[k][0] = (u3[k][4])/(u3[k][4]+u3[k][0]-u3[k][1])
    yuzhi[k][1] = (u3[k][5]-u3[k][4])/(u3[k][1]+u3[k][5]-u3[k][4])
yuzhi    
#%%

#%%

#%%
# 计算不同对象的聚合相对效用函数矩阵
X = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
theta = 0.5
n, m = len(data), len(data.T)
lamda = np.zeros((len(X),6));
u = np.ones((len(X),6));
u1 = np.zeros((len(X),6));
u3 = np.zeros((len(X),6))
x = data.iloc[1]
yuzhi = np.zeros((len(X),2))

for k in range(0,len(X)):
    singam = X[k]
   
    # 计算相对损失函数
    lamda[k,1] = singam * x[0];
    lamda[k,2] = x[0];
    lamda[k,3] = 1-x[0];
    lamda[k,4] = singam * (1-x[0]);
    
    # 计算基于后悔理论的效用函数
    u[k,1] = 1 - (1 - np.exp(-theta * lamda[k,1]))/theta
    u[k,2] = 1 - (1 - np.exp(-theta * lamda[k,2]))/theta
    u[k,3] = 1 - (1 - np.exp(-theta * lamda[k,3]))/theta
    u[k,4] = 1 - (1 - np.exp(-theta * lamda[k,4]))/theta
        
    # 计算相对效用函数
    u1[k,0] = 1 - u[k,2] 
    u1[k,1] = u[k,1] - u[k,2] 
    u1[k,4] = u[k,4] - u[k,3] 
    u1[k,5] = 1 - u[k,3]
    u2 = np.max(u1, axis = 0)
    delta = 0.1
    for j in range(0,6):
        u3[k][j] = u1[k,j] + 1 - np.exp(-delta * (u1[k,j]-u2[j]))
            
    #根据基于后悔理论的相对效用函数计算不同对象的不同粒度下的阈值
    yuzhi[k][0] = (u3[k][4])/(u3[k][4]+u3[k][0]-u3[k][1])
    yuzhi[k][1] = (u3[k][5]-u3[k][4])/(u3[k][1]+u3[k][5]-u3[k][4])
yuzhi    
#%%
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 16,
         }
config = {
            "font.family": 'serif',
            "font.size": 14,# 相当于小四大小
            "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            "font.serif": ['SimSun'],#宋体
            'axes.unicode_minus': False # 处理负号，即-号
         }
rcParams.update(config)
#%%
print(yuzhi.T[:][0])
print(yuzhi.T[:][1])
#%%
# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (6.0, 4.0)
rc('font',size=16); rc('text', usetex=True)  #调用tex字库

X = ('0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45') 
Y1 = [0.9534727,  0.90170389, 0.84930223, 0.79628486, 0.74266977, 0.68847576, 0.63372244, 0.57843022, 0.55262027]
Y2 = [0.06763845, 0.13910823, 0.20776334, 0.27379122, 0.33736312, 0.39863589, 0.45775336, 0.51484774, 0.52004075]

bar_width = 0.3 # 条形宽度
index_male = np.arange(len(X)) # 阈值1条形图的横坐标
index_female = index_male + bar_width # 阈值2条形图的横坐标
 
# 使用两次 bar 函数画出两组条形图
plt.bar(index_male, height=Y1, width=bar_width, color='royalblue', label=r'$\alpha_1$')
plt.bar(index_female, height=Y2, width=bar_width, color='coral', label=r'$\beta_1$')

plt.xticks(index_male + bar_width/2, X)
plt.xlabel('The value of $\sigma$',font1)
plt.ylabel('The value of thresholds',font1)
plt.legend() # 显示图例
savefig("参数实验1.1.jpg", dpi=300, bbox_inches='tight')
plt.show()
#%%
# 计算不同对象的聚合相对效用函数矩阵
X = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
singam = 0.15
n, m = len(data), len(data.T)
lamda = np.zeros((len(X),6));
u = np.ones((len(X),6));
u1 = np.zeros((len(X),6));
u3 = np.zeros((len(X),6))
x = data.iloc[1]
yuzhi = np.zeros((len(X),2))

for k in range(0,len(X)):
    theta = X[k]
   
    # 计算相对损失函数
    lamda[k,1] = singam * x[0];
    lamda[k,2] = x[0];
    lamda[k,3] = 1-x[0];
    lamda[k,4] = singam * (1-x[0]);
    
    # 计算基于后悔理论的效用函数
    u[k,1] = 1 - (1 - np.exp(-theta * lamda[k,1]))/theta
    u[k,2] = 1 - (1 - np.exp(-theta * lamda[k,2]))/theta
    u[k,3] = 1 - (1 - np.exp(-theta * lamda[k,3]))/theta
    u[k,4] = 1 - (1 - np.exp(-theta * lamda[k,4]))/theta
        
    # 计算相对效用函数
    u1[k,0] = 1 - u[k,2] 
    u1[k,1] = u[k,1] - u[k,2] 
    u1[k,4] = u[k,4] - u[k,3] 
    u1[k,5] = 1 - u[k,3]
    u2 = np.max(u1, axis = 0)
    delta = 0.1
    for j in range(0,6):
        u3[k][j] = u1[k,j] + 1 - np.exp(-delta * (u1[k,j]-u2[j]))
            
    #根据基于后悔理论的相对效用函数计算不同对象的不同粒度下的阈值
    yuzhi[k][0] = (u3[k][4])/(u3[k][4]+u3[k][0]-u3[k][1])
    yuzhi[k][1] = (u3[k][5]-u3[k][4])/(u3[k][1]+u3[k][5]-u3[k][4])
yuzhi    
#%%
print(yuzhi.T[:][0])
print(yuzhi.T[:][1])
#%%
yuzhi[0][0]-yuzhi[8][0]
#%%
0.8720 - 0.8419
#%%
X = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
Y1 = [0.87197793, 0.86847267, 0.86489605, 0.86124738, 0.857526,   0.85373122, 0.84986238, 0.84591882, 0.8418999 ]
Y2 = [0.182308,   0.18584736, 0.1894313,  0.19305984, 0.19673299, 0.20045076, 0.20421313, 0.20802009, 0.21187163]

plt.rcParams['figure.figsize'] = (6.0, 4.0)
bar_width = 0.3 # 条形宽度
index_male = np.arange(len(X)) # 阈值1条形图的横坐标
index_female = index_male + bar_width # 阈值2条形图的横坐标
# 使用两次 bar 函数画出两组条形图
plt.bar(index_male, height=Y1, width=bar_width, color='royalblue', label=r'$\alpha_1$')
plt.bar(index_female, height=Y2, width=bar_width, color='coral', label=r'$\beta_1$')
plt.xticks(index_male + bar_width/2, X)
plt.xlabel('The value of 'r'$\theta$',font1)
plt.ylabel('The value of thresholds',font1)
plt.legend() # 显示图例
plt.title('(a)', y=-0.3)
# savefig("参数实验2.2.jpg", dpi=300, bbox_inches='tight')
plt.show()
#%%
plt.rcParams['figure.figsize'] = (3.0, 2.0)
X = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
plt.plot(X, Y1, color='royalblue', label=r'$\alpha_1$')
plt.legend() # 显示图例
savefig("参数实验3.3.jpg", dpi=300, bbox_inches='tight')
plt.show()
#%%
plt.rcParams['figure.figsize'] = (3.0, 2.0)
plt.plot(X, Y2, color='coral', label=r'$\beta_1$')
plt.legend() # 显示图例
savefig("参数实验4.4.jpg", dpi=300, bbox_inches='tight')
plt.show()
#%%
X = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
Y1 = [0.87197793, 0.86847267, 0.86489605, 0.86124738, 0.857526,   0.85373122, 0.84986238, 0.84591882, 0.8418999 ]
Y2 = [0.182308,   0.18584736, 0.1894313,  0.19305984, 0.19673299, 0.20045076, 0.20421313, 0.20802009, 0.21187163]
plt.rcParams['figure.figsize'] = (3, 4.0)
a = subplot(2,1,1)
plt.plot(X, Y1, color='royalblue', label=r'$\alpha_1$')
plt.legend() # 显示图例
plt.xticks([0.5,0.9]) #设置图片横坐标的显示
b = subplot(2,1,2)
b.plot(X, Y2, color='coral', label=r'$\beta_1$')
plt.legend() # 显示图例
plt.xticks([0.5,0.9])
plt.title('(b)', y=-0.4)
savefig("参数实验7.7.jpg", dpi=300, bbox_inches='tight')
plt.show()
#%%
# 计算不同对象的聚合相对效用函数矩阵
X = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
singam = 0.15
theta = 0.5
n, m = len(data), len(data.T)
lamda = np.zeros((1,6));
u = np.ones((1,6));
u1 = np.zeros((1,6));
u3 = np.zeros((len(X),6))
x = data.iloc[1]
yuzhi = np.zeros((len(X),2))

for k in range(0,len(X)):
    # 计算相对损失函数
    lamda[0,1] = singam * x[0];
    lamda[0,2] = x[0];
    lamda[0,3] = 1-x[0];
    lamda[0,4] = singam * (1-x[0]);
    
    # 计算基于后悔理论的效用函数
    u[0,1] = 1 - (1 - np.exp(-theta * lamda[0,1]))/theta
    u[0,2] = 1 - (1 - np.exp(-theta * lamda[0,2]))/theta
    u[0,3] = 1 - (1 - np.exp(-theta * lamda[0,3]))/theta
    u[0,4] = 1 - (1 - np.exp(-theta * lamda[0,4]))/theta
        
    # 计算相对效用函数
    u1[0,0] = 1 - u[0,2] 
    u1[0,1] = u[0,1] - u[0,2] 
    u1[0,4] = u[0,4] - u[0,3] 
    u1[0,5] = 1 - u[0,3]
    u2 = np.max(u1, axis = 0)
    delta = X[k]
    for j in range(0,6):
        u3[k][j] = u1[0,j] + 1 - np.exp(-delta * (u1[0,j]-u2[j]))
            
    #根据基于后悔理论的相对效用函数计算不同对象的不同粒度下的阈值
    yuzhi[k][0] = (u3[k][4])/(u3[k][4]+u3[k][0]-u3[k][1])
    yuzhi[k][1] = (u3[k][5]-u3[k][4])/(u3[k][1]+u3[k][5]-u3[k][4])
yuzhi    
#%%
print(yuzhi.T[:][0])
print(yuzhi.T[:][1])
#%%
0.1967 - 0.2110 
#%%
X = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
Y1 = yuzhi.T[:][0]
Y2 = yuzhi.T[:][1]
bar_width = 0.3 # 条形宽度
index_male = np.arange(len(X)) # 阈值1条形图的横坐标
index_female = index_male + bar_width # 阈值2条形图的横坐标
plt.rcParams['figure.figsize'] = (6.0, 4.0)
# 使用两次 bar 函数画出两组条形图
plt.bar(index_male, height=Y1, width=bar_width, color='royalblue', label=r'$\alpha_1$')
plt.bar(index_female, height=Y2, width=bar_width, color='coral', label=r'$\beta_1$')

plt.xticks(index_male + bar_width/2, X)
plt.xlabel('The value of 'r'$\delta$',font1)
plt.ylabel('The value of thresholds',font1)
plt.legend() # 显示图例
plt.title('(a)', y=-0.3)
savefig("参数实验5.5.jpg", dpi=300, bbox_inches='tight')
plt.show()
#%%
X = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
Y1 = yuzhi.T[:][0]
Y2 = yuzhi.T[:][1]
plt.rcParams['figure.figsize'] = (3.0, 4.0)
a = subplot(2,1,1)
plt.plot(X, Y1, color='royalblue', label=r'$\alpha_1$')
plt.legend() # 显示图例
b = subplot(2,1,2)
b.plot(X, Y2, color='coral', label=r'$\beta_1$')
plt.legend() # 显示图例
plt.title('(b)', y=-0.4)
savefig("参数实验6.6.jpg", dpi=300, bbox_inches='tight')
plt.show()
#%%
# 计算不同对象的聚合相对效用函数矩阵
X1 = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
X2 = [0.1, 0.5, 0.9]
n, m = len(data), len(data.T)
x = data.iloc[1]

for k1 in range(0,len(X1)):
    for k2 in range(0,len(X2)):
        singam = X1[k1]
        theta = X2[k2]
        u = np.ones((1,6));
        u1 = np.zeros((1,6));
        u3 = np.zeros((1,6))
        
        # 计算相对损失函数
        lamda[0,1] = singam * x[0];
        lamda[0,2] = x[0];
        lamda[0,3] = 1-x[0];
        lamda[0,4] = singam * (1-x[0]);
    
        # 计算基于后悔理论的效用函数
        u[0,1] = 1 - (1 - np.exp(-theta * lamda[0,1]))/theta
        u[0,2] = 1 - (1 - np.exp(-theta * lamda[0,2]))/theta
        u[0,3] = 1 - (1 - np.exp(-theta * lamda[0,3]))/theta
        u[0,4] = 1 - (1 - np.exp(-theta * lamda[0,4]))/theta
        
        # 计算相对效用函数
        u1[0,0] = 1 - u[0,2] 
        u1[0,1] = u[0,1] - u[0,2] 
        u1[0,4] = u[0,4] - u[0,3] 
        u1[0,5] = 1 - u[0,3]
        u2 = np.max(u1, axis = 0)
        delta = 0.1
        for j in range(0,6):
            u3[0][j] = u1[0,j] + 1 - np.exp(-delta * (u1[0,j]-u2[j]))
        
        yuzhi = np.zeros((1,6))
        #根据基于后悔理论的相对效用函数计算不同对象的不同粒度下的阈值
        yuzhi[0][0] = (u3[0][4])/(u3[0][4]+u3[0][0]-u3[0][1])
        yuzhi[0][1] = (u3[0][5]-u3[0][4])/(u3[0][1]+u3[0][5]-u3[0][4])
        print(singam,theta)
        print((round(yuzhi[0][0],4),round(yuzhi[0][1],4)))
#%%
# 计算不同对象的聚合相对效用函数矩阵
X1 = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
X2 = [0.1, 0.5, 1]
n, m = len(data), len(data.T)
x = data.iloc[1]
theta = 0.5
for k1 in range(0,len(X1)):
    for k2 in range(0,len(X2)):
        singam = X1[k1]
        u = np.ones((1,6));
        u1 = np.zeros((1,6));
        u3 = np.zeros((1,6))
        
        # 计算相对损失函数
        lamda[0,1] = singam * x[0];
        lamda[0,2] = x[0];
        lamda[0,3] = 1-x[0];
        lamda[0,4] = singam * (1-x[0]);
    
        # 计算基于后悔理论的效用函数
        u[0,1] = 1 - (1 - np.exp(-theta * lamda[0,1]))/theta
        u[0,2] = 1 - (1 - np.exp(-theta * lamda[0,2]))/theta
        u[0,3] = 1 - (1 - np.exp(-theta * lamda[0,3]))/theta
        u[0,4] = 1 - (1 - np.exp(-theta * lamda[0,4]))/theta
        
        # 计算相对效用函数
        u1[0,0] = 1 - u[0,2] 
        u1[0,1] = u[0,1] - u[0,2] 
        u1[0,4] = u[0,4] - u[0,3] 
        u1[0,5] = 1 - u[0,3]
        u2 = np.max(u1, axis = 0)
        delta = X2[k2]
        for j in range(0,6):
            u3[0][j] = u1[0,j] + 1 - np.exp(-delta * (u1[0,j]))
        
        yuzhi = np.zeros((1,6))
        #根据基于后悔理论的相对效用函数计算不同对象的不同粒度下的阈值
        yuzhi[0][0] = (u3[0][4])/(u3[0][4]+u3[0][0]-u3[0][1])
        yuzhi[0][1] = (u3[0][5]-u3[0][4])/(u3[0][1]+u3[0][5]-u3[0][4])
        print(singam,delta)
        print((round(yuzhi[0][0],4),round(yuzhi[0][1],4)))
#%%
