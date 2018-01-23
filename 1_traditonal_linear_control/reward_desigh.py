
"""
Reward Design
Designer: Lin Cheng  2018.01.22
aa
aaaaa
aaa
"""

import numpy as np
import matplotlib.pyplot as plt

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
        y[i] = -np.log2(1.000000001 - xxx) / np.log2(b) / 100

plt.figure(1)
plt.plot(x, y)
plt.show()





