## simulations
import numpy as np
import input_parameters as cont
import differentiation_matrix as diff

## initialize Chebyshev differentiation matrix
x,y = diff.xy_grid()

## D and D transpose
D = diff.D_matrix(x)
TD = np.transpose(D)

## D2 and D2 transpose
D2 = D*D
TD2 = TD*TD

## boundary conditions

## D
D00 = D[0,0]
Dm0 = D[-1,0]
D0m = D[0,-1]
Dmm = D[-1,-1]

D0k = D[0,1:-1]
Dmk = D[-1,1:-1]

## D transpose
TD00 = TD[0,0]
TD0m = TD[0,-1]
TDm0 = TD[-1,0]
TDmm = TD[-1,-1]

TDk0 = TD[1:-1,0]
TDkm = TD[1:-1,-1]

## NLT
def NL(U,V,X):
    NLX = cont.asp* U * (D*X) + cont.asp* V * (X*np.transpose(D))
    return NLX

## Step 1 Temperature
def RHS_T(U0,V0,T0,U1,V1,T1):

    RHST = NL(U1,V1,T1) - NL(U0,V0,T0) + ((4*T1 + T0)/2*cont.dt)

    for i in range(1,cont.M):
        for j in range(1,cont.M):
            RHST[i,j] -= cont.asp*cont.asp*(T1[i,0]*TD2[0,j] + T1[i,-1]*TD2[-1,j])

    return RHST

'''
def LHS_T():

    ## Helmoltz constants
    A = np.empty_like(D2[1:-1,1:-1])
    for i in range(1,cont.M):
        A[i-1,:] = cont.asp*cont.asp*(D2[i,1:-1] - D2[i,0] * ((Dmm*D0k - D0m*Dmk)/(D00*Dmm - Dm0*D0m)) - D2[i,-1]*((Dm0*D0k - D00*Dmk)/(Dm0*D0m - D00*Dmm)))
    B = cont.asp*cont.asp*TD2[1:-1,1:-1]
    C = 3*(cont.Pr/(2*cont.dt))

    return A,B,C
'''

def eigens_T():
    eigen_vec_T1  = np.zeros([cont.M+1,cont.M+1])
    eigen_val_T1 = np.zeros([cont.M+1])
    eigen_vec_T2  = np.zeros([cont.N+1,cont.N+1])
    eigen_val_T2 = np.zeros([cont.N+1])
    tempx = np.zeros([cont.M+1,cont.M+1])
    tempy = np.zeros([cont.N+1,cont.N+1])
    D00_MM_DM0_0M   = D[0,0]*D[cont.M,cont.M]   - D[cont.M,0]*D[0,cont.M]
    TD00_NN_TD0N_N0 = TD[0,0]*TD[cont.N,cont.N] - TD[0,cont.N]*TD[cont.N,cont.N] 
    for i in range(1,cont.M): # Neumann boundary conditions
        tempx[i,1:cont.M] =-D2[i,0] * (D[cont.M,cont.M]*D[0,1:cont.M]-D[0,cont.M]*D[cont.M,1:cont.M])/D00_MM_DM0_0M \
                    +D2[i,cont.M] * (D[cont.M,0]*D[0,1:cont.M]-D[0,0]*D[cont.M,1:cont.M])/D00_MM_DM0_0M 
    tempx[1:cont.M,1:cont.M] = cont.asp**2 * (D2[1:cont.M,1:cont.M] - tempx[1:cont.M,1:cont.M])
                       # Dirichlet boundary conditions 
    tempy[1:cont.N,1:cont.N] = cont.asp**2 * TD2[1:cont.N,1:cont.N] 
    eigen_val_T1[1:cont.M], eigen_vec_T1[1:cont.M,1:cont.M] = np.linalg.eig(tempx[1:cont.M,1:cont.M])
    eigen_val_T2[1:cont.N]  , eigen_vec_T2[1:cont.N,1:cont.N] = np.linalg.eig(tempy[1:cont.N,1:cont.N] )

    return eigen_vec_T1, eigen_val_T1, eigen_vec_T2, eigen_val_T2

## Step 2 Pressure
def RHS_P(U0,V0,T0,U1,V1,T1,T2):

    RHSP1 = cont.asp*D*(-2*NL(U1,V1,U1) + NL(U0,V0,U0))
    RHSP2 = cont.asp*((-2*NL(U1,V1,V1) + NL(U0,V0,V0) + cont.Ri*cont.Re*cont.Re*T2)*TD)

    RHSP = np.empty_like(D2[1:-1,1:-1])
    for i in range(1,cont.M):
        for j in range(1,cont.M):
            RHSP[i-1,j-1] = RHSP1[i,j] + RHSP2[i,j] - cont.asp*(D2[i,0]*((RHSP1[0,j]*Dmm -  RHSP1[-1,j]*D0m)/(D00*Dmm - D0m*Dm0)) + D2[i,-1]*((RHSP1[0,j]*Dm0 -  RHSP1[-1,j]*D00)/(Dm0*D0m - D00*Dmm)) + TD2[0,j]*((RHSP2[i,0]*TDmm - RHSP2[i,-1]*TDm0)/(TD00*TDmm - TD0m*TDm0)) + TD2[-1,i]*(RHSP2[i,0]*TD0m - RHSP2[i,-1]*TD00)/(TD0m*TDm0 - TD00*TDmm))

    return RHSP

'''
def LHS_P():

    ## Helmoltz constants
    A = np.empty_like(D2[1:-1,1:-1])
    B = np.empty_like(D2[1:-1,1:-1])
    for i in range(1,cont.M-1):
        A[i-1,:] = cont.asp*cont.asp*( D2[i,1:-1] - D2[i,-1]* ((Dm0*D0k - D00*Dmk)/(Dm0*D0m - D00*Dmm)) - D2[i,0] * ((Dmm*D0k - D0m*Dmk)/(D00*Dmm - Dm0*D0m))  )
        B[:,i-1] = cont.asp*cont.asp*( TD2[1:-1,i] - ((TDmm*TDk0 - TDm0*TDkm)/(TD00*TDmm - TD0m*TDm0)) * TD2[0,i] - ((TD0m*TDk0 - TD00*TDkm)/(TD0m*TDm0 - TD00*TDmm))*TD2[-1,i]  )
    C = 0

    return A,B,C
'''

## step 3
def RHS_U(U0,V0,T0,U1,V1,T1,T2,P):

    RHSU = cont.asp*D*P + 2*NL(U1,V1,U1) - NL(U0,V0,U0) + ((4*U1 - U0)/2*cont.dt) - cont.Ri*cont.Re*cont.Re*T2
    RHSV = cont.asp*P*TD + NL(U1,V1,V1) - NL(U0,V0,V0) + ((4*V1 + V0)/2*cont.dt) - cont.Ri*cont.Re*cont.Re*T2

    for i in range(1,cont.M):
        for j in range(1,cont.M):
            RHSU[i,j] -= cont.asp*cont.asp*(D2[i,0]*U0j[0,j] + D2[i,-1]*Umj[0,j] + Ui0[i,0]*TD2[0,j] + Uim[i,0]*TD2[-1,j])
            RHSV[i,j] -= cont.asp*cont.asp*(D2[i,0]*V0j[0,j] + D2[i,-1]*Vmj[0,j] + Vi0[i,0]*TD2[0,j] + Vim[i,0]*TD2[-1,j])

    return RHSU, RHSV


def LHS_U():

    ## Helmoltz constants
    A = D2[1:-1,1:-1]
    B = TD2[1:-1,1:-1]
    C = 3*(cont.Pr/(2*cont.dt))

    return A,B,C


## step 4
def RHS_Z(U, V):

    RHSZ = cont.asp*(D*U + V*TD)
    RHSZ *= 0.25

    return RHSZ

def LHS_Z():

    ## Helmoltz constants
    A = np.empty_like(D2[1:-1,1:-1])
    B = np.empty_like(D2[1:-1,1:-1])
    ## Helmoltz constants
    for i in range(1,cont.M):
        A[i-1,:] = ( D2[i,1:-1] - D2[i,-1] * ((Dm0*D0k - D00*Dmk)/(Dm0*D0m - D00*Dmm)) - D2[i,0] * ((Dmm*D0k - D0m*Dmk)/(D00*Dmm - Dm0*D0m))  )
        B[:,i-1] = ( TD2[1:-1,i] - ((TDmm*TDk0 - TDm0*TDkm)/(TD00*TDmm - TD0m*TDm0)) * TD2[0,i] - ((TD0m*TDk0 - TD00*TDkm)/(TD0m*TDm0 - TD00*TDmm))*TD2[-1,i]  )
    C = 0
    return A,B,C

## step 5
def PUV(P, U, V, Z):
    TD = np.transpose(D)
    P2 = P + (3*(cont.Pr/(2*cont.dt))) * Z
    U2 = U - cont.asp * D * Z
    V2 = V  - cont.asp * Z * TD

    return P2, U2, V2
