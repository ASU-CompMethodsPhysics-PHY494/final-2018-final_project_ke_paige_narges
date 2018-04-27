import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## plots.py

def XYGrid(x,y):
    xx,yy = np.meshgrid(x, y)
    plt.plot(xx,yy, marker='.', linestyle='none')
    plt.show()
