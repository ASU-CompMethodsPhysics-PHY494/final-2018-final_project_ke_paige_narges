import numpy as np
import input_parameters as cont
## initialize_matrix
m = cont.M
n = cont.N
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
    D[n,n] =  -D[0,0]

    for d in i:
        for e in j:
            if d != e:
                D[e,d] = (ci[e]/cj[d]) * ((-1)**(d+e))/(2*x[e] - 2*x[d])
               
    diag = np.sum(D[1:-1],axis = 1)
    D[1:-1, 1:-1] = D[1:-1, 1:-1] - np.diagflat(diag)

    D1 = D
    return D1

def T_0():
    T0 = np.zeros((m+1,n+1))
    T0 = np.asmatrix(T0)
    T0[:,0] =  0.5
    T0[:,-1] = -0.5
    return T0

def U_0(x): # only top boundary is moving
    U0 = np.zeros((m+1,n+1))
    U0 = np.asmatrix(U0)
    f = 1 - np.exp( -2*(1-(2*x)**2)/0.06 )
    U0[:,0] =  np.transpose(np.asmatrix(cont.Re*f))
    return U0

def V_0():
    V0 = np.zeros((m+1,n+1))
    V0 = np.asmatrix(V0)
    return V0
