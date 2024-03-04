import dataset
from matplotlib import pyplot as plt
import numpy as np
m = 100
xs, ys = dataset.get_beans(m)
plt.title("Size-Toxicity Function", fontsize=12)
plt.xlabel("Size")
plt.ylabel("Toxicity")

# Stochastic Gradient Descent
# AKA batch size of 1
w = 0.1
b = 0.1
alpha = 0.1

for _ in range(1000):
    dw = 2/m * (w * np.sum(xs**2) + b * np.sum(xs) - np.sum(xs * ys))
    db = 2/m * (m * b + w * np.sum(xs) - np.sum(ys))
    
    w = w - alpha * dw
    b = b - alpha * db
    plt.clf()
    plt.scatter(xs, ys)
    plt.xlim(0, 1)
    plt.ylim(0, 1.2)
    y_pre = w * xs + b
    plt.plot(xs, y_pre)
    plt.pause(0.001)

plt.show()
