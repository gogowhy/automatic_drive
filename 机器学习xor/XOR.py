import numpy as np
import matplotlib.pyplot as plt
# 输出图像中汉字字体相关的设置
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

def sigmoid(x):    # 激活函数sigmoid
    y = 1 / (1 + np.exp(-x))
    return y

##### 1.初始设置
# 原始数据
x_raw = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
y_raw = np.array([0 ,1, 1, 0])
# 初始权值（大家可以尝试使用不同的初始权值，查看迭代轮数和收敛效果，此处权值也可使用random来生成）
w_1 = np.array([[1.0, 1.0],
                [-1.0, -1.0],
                [0.0, 0.0]])     # 第一层初始权值

w_2 = np.array([[1.0],
                [-1.0],
                [0.0]])     # 第二层初始权值
# 学习率
lr = 0.6
# 初始迭代误差
err = 0.5
j = 0
E = []
Y_1 = []
Y_2 = []
Y_3 = []
Y_4 = []
# 变量初始化
x_1 = np.c_[x_raw, [1, 1, 1, 1]]
x_2 = np.zeros([4, 3])
y = np.array([0.0, 0.0, 0.0, 0.0])

##### 2.迭代训练
while err >= 0.008:    # 当误差大于0.008时继续迭代训练
    err = 0
    for i in range(4):
        # 前向信号传递
        x_2[i, 0] = sigmoid(np.matmul(x_1[i, :], w_1[:, 0]))
        x_2[i, 1] = sigmoid(np.matmul(x_1[i, :], w_1[:, 1]))
        x_2[i, 2] = 1
        y[i] = sigmoid(np.matmul(x_2[i, :], w_2))
        err = err + 0.5 * np.square(y[i] - y_raw[i])
        # 反向误差传递，权值更新
        delta_3 = (y[i] - y_raw[i]) * y[i] * (1 - y[i])
        delta_21 = delta_3 * w_2[0] * x_2[i, 0] * (1 - x_2[i, 0])
        delta_22 = delta_3 * w_2[1] * x_2[i, 1] * (1 - x_2[i, 1])
        w_2[0] = w_2[0] - lr * delta_3 * x_2[i, 0]
        w_2[1] = w_2[1] - lr * delta_3 * x_2[i, 1]
        w_2[2] = w_2[2] - lr * delta_3 * x_2[i, 2]
        w_1[0, 0] = w_1[0, 0] - lr * delta_21 * x_1[i, 0]
        w_1[1, 0] = w_1[1, 0] - lr * delta_21 * x_1[i, 1]
        w_1[2, 0] = w_1[2, 0] - lr * delta_21 * x_1[i, 2]
        w_1[0, 1] = w_1[0, 1] - lr * delta_22 * x_1[i, 0]
        w_1[1, 1] = w_1[1, 1] - lr * delta_22 * x_1[i, 1]
        w_1[2, 1] = w_1[2, 1] - lr * delta_22 * x_1[i, 2]
    j = j + 1    # 迭代轮数+1
    Y_1.append(y[0])    # 00 样本在每一轮训练中的输出
    Y_2.append(y[1])    # 01
    Y_3.append(y[2])    # 10
    Y_4.append(y[3])    # 11
    E.append(err)    # 每一轮的迭代误差
    
##### 3.结果输出
print('\n输入组合  期望输出  实际输出\n')
for i in range(4):
    print('  %d %d      %d     %f\n' % (x_raw[i, 0], x_raw[i, 1], y_raw[i], y[i]))
print('\n迭代 %d 轮结束，最终误差为 %f\n' % (j, err))

Y_1 = np.array(Y_1)
Y_2 = np.array(Y_2)
Y_3 = np.array(Y_3)
Y_4 = np.array(Y_4)
E = np.array(E)
i = np.arange(0, j, 1)

plt.figure()
plt.plot(i, E[i])
plt.xlabel('迭代次数', fontsize = 10)
plt.ylabel('误差', fontsize = 10)
plt.title('误差随迭代次数变化曲线', fontsize = 10)
plt.show()

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(i, Y_1[i])
plt.xlabel('迭代次数', fontsize = 10)
plt.ylabel('00样本输出', fontsize = 10)
plt.subplot(2, 2, 2)
plt.plot(i, Y_2[i])
plt.xlabel('迭代次数', fontsize = 10)
plt.ylabel('01样本输出', fontsize = 10)
plt.subplot(2, 2, 3)
plt.plot(i, Y_3[i])
plt.xlabel('迭代次数', fontsize = 10)
plt.ylabel('10样本输出', fontsize = 10)
plt.subplot(2, 2, 4)
plt.plot(i, Y_4[i])
plt.xlabel('迭代次数', fontsize = 10)
plt.ylabel('11样本输出', fontsize = 10)
plt.show()