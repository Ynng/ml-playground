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

for _ in range(100):
    # error = (w * x - y)^2 = w^2 * x^2 - 2wxy + y^2
    
    k = 2 * w * np.sum(xs**2) - 2 * np.sum(xs * ys)
    k /= 100
    alpha = 0.1
    w = w - alpha * k
    plt.clf()
    plt.scatter(xs, ys)
    plt.xlim(0, 1)
    plt.ylim(0, 1.2)
    y_pre = w * xs
    plt.plot(xs, y_pre)
    plt.pause(0.01)
