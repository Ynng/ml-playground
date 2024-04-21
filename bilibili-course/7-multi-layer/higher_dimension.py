import numpy as np
import dataset
import plot_utils

m = 100
xs,ys = dataset.get_beans(m)
print(xs)
print(ys)

plot_utils.show_scatter(xs,ys)
