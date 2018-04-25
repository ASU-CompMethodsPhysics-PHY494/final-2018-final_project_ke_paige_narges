import numpy as np
import constants as cont
## initialize_matrix
m = cont.m
n = cont.n
i = np.arange(0,m+1)
j = np.arange(0,n+1)

def xy_grid():


    x = []
    y = []
    for d in i:
        xi = 0.5* np.cos(np.pi/m * i[d])
        yi = 0.5 *np.cos(np.pi/n * j[d])
        x.append(xi)
        y.append(yi)

    x = np.asarray(x)
    y = np.asarray(y)

    return x,y

def D_matrix(x):
    ci = np.ones(m+1)
    ci[0] = ci[-1] = 2

    cj = np.ones(n+1)
    cj[0] = cj[-1] = 2

    cj = np.asarray(cj)
    ci = np.asarray(ci)

    D = np.zeros((m+1,n+1))
    D = np.asmatrix(D)
    D[0,0] =  (2 * n**2 + 1)/6
    D[n,n] =  (2 * n**2 + 1)/6

    for d in i:
        for e in j:
            if d != e:
                D[d,e] = (ci[d]*(-1)**(d+e))/(cj[e]*(x[d] - x[e]))

    diag = np.sum(D[1:-1],axis = 1)
    D[1:-1, 1:-1] = D[1:-1, 1:-1] - np.diagflat(diag)

    D1 = D
    return D1


def T_0(top = 0 , bot = 0):
    T0 = np.zeros((m+1,n+1))
    T0 = np.asmatrix(T0)
    T0[0,:] =  top
    T0[-1,:] = bot
    return T0

def U_0(top = 0 , bot = 0):
    U0 = np.zeros((m+1,n+1))
    U0 = np.asmatrix(U0)
    U0[0,:] =  top
    U0[-1,:] = bot
    return U0

def V_0(top = 0 , bot = 0):
    V0 = np.zeros((m+1,n+1))
    V0 = np.asmatrix(V0)
    V0[0,:] =  top
    V0[-1,:] = bot
    return V0
