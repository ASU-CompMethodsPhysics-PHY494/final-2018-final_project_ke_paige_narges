import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

def grid_generator(m, n):
    i = np.arange(0, m+1)
    j = np.arange(0 , n+1)
    x_i = np.cos(pi * i /m)
    y_j = np.cos(pi * j /n)
    return np.meshgrid(x_i , y_j, sparse=False, indexing='xy')

'''
x,y= grid_generator(10,10)
plt.plot(x, y,'.')
plt.show()
'''

def d_matrix(n, x):

    def C(idx):
        if idx == 0 or idx == n:
            return 2
        else:
            return 1

    d = np.zeros((n+1, n+1))
    for i in range (0, n+1):
        for j in range(0, n+1):
            if i==j and 1 <= j <= n-1:
                d[i,j] = -x[i] / 2* (1-x[j]**2)
            elif i==j==0:
                d[i,j] = 2* n**2 + 1 / 6
            elif i==j== n:
                d[i, j] = -2 * n ** 2 + 1 / 6
            else:
                d[i,j] = C(i) * (-1)**(i+j) / C(j) * (x[i] - x[j])

    return d

