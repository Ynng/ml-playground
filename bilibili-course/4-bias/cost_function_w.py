import dataset
import matplotlib.pyplot as plt
import numpy as np

# This script plots the cost function for different values of w and b in a 3D plot

# Getting the dataset
m = 100
xs, ys = dataset.get_beans(m)

# A plot of the data
plt.title("Size-Toxicity Function", fontsize=12)
plt.xlabel("Bean Size")
plt.ylabel("Toxicity")
plt.xlim(0, 1)
plt.ylim(0, 1.5)

plt.scatter(xs, ys)

# Plot a random line for sanity check
w = 0.1
b = 0.1
y_pre = w * xs + b
plt.plot(xs, y_pre)
plt.show()

# Actually plotting the 3D plot

# Setup the 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_zlim(0,2)

# Setup all the possible values for w and b
ws = np.arange(-1, 2, 0.1)
bs = np.arange(-2, 2, 0.1)

# plot a curve for each value of b
for b in bs:
    es = []
    for w in ws:
        y_pre = w * xs + b
        e = np.sum((ys - y_pre) ** 2) / m
        es.append(e)
    ax.plot(ws, es, zs=b, zdir='y')

# Setting labels
ax.set_xlabel('Weight w')
ax.set_ylabel('Error e')
ax.set_zlabel('Bias b')
plt.show()
