
"""
Reward Design
Designer: Lin Cheng  2018.01.22
aa
aaaaa
aaa
"""

import numpy as np
import matplotlib.pyplot as plt


# 第一种饱和惩罚函数
x = np.linspace(0, 1, 100000)

y = np.zeros(x.shape)

b = 0.99

a = x.shape

for i in range(0, 100000):
    x_penalty = 0.75
    if x[i] < x_penalty:
        y[i] = 0
    else:
        xxx = (x[i] - x_penalty)/(1 - x_penalty)
        y[i] = - np.log2(1.01 - xxx) / np.log2(b) / 3


# 第二种饱和惩罚函数
x2 = np.linspace(0, 5, 100000)
y2 = np.zeros(x2.shape)
for i in range(0, 100000):
    if x2[i] < x_penalty:
        y2[i] = 0
    else:
        # xxx = (x[i] - x_penalty) / (1 - x_penalty)
        y2[i] = - 1000* (np.exp(0.1*(x2[i]-x_penalty)) -1) / 2



# 图形显示
plt.figure(1)
plt.plot(x, y)

plt.figure(2)
plt.plot(x2, y2)
plt.show()





