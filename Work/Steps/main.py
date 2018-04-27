## Import necessary packages
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
## only in jupyte notebook
##%matplotlib notebook

# Import associated files.
import Simulations
import plots
import constants as cont
import initialze_matrix


## main.py

## View XYGrid
x,y = initialze_matrix.xy_grid()
plots.XYGrid(x,y)

## Call initialized U,V,T matrices
U0 = initialze_matrix.U_0( top = cont.re)
U1 = U0
V0 = initialze_matrix.V_0()
V1 = V0
T0 = initialze_matrix.T_0(top = cont.a, bot = -cont.a)
T1 = T0

## NOT TRUE
T2 = T0

## Call Chebyshev differential matrix
D = initialze_matrix.D_matrix(x)

## T boundary conditions
Tm0 = T1[:,0]
Tmm = T1[:,-1]


## Call RHS of temporal scheme
print(Simulations.RHS_T(U1,V1,T1,U0,V0,T0,D,Tm0,Tmm))


##
#for jt in range(2,Nt):

#    T2[1:-1] =
#    T0[1:-1] = T1[1:-1]
#    T1[1:-1] = T2[1:-1]
