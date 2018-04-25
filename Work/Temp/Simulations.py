## simulations
import numpy as np
import constants as cont

## NLT
def NL_T(U,V,T,D):
    NLT = cont.asp* U * (D*T) + cont.asp* V * (T*np.transpose(D))
    return NLT

## right hand side of Temportal Scheme
def RHS_T(U1,V1,T1,U0,V0,T0,D,Tm0,Tmm):
    D2 = D*D
    tD2 = np.transpose(D2)
    tD2mj = tD2[-1:]
    tD20j = tD2[0,:]
    RHST = NL_T(U1,V1,T1,D) - NL_T(U0,V0,T0,D) + ((4*T1 + T0)/2*cont.h)
    RHST -= (Tm0*tD20j  + Tmm*tD2mj)
    return RHST

## left hand side of Temporal Scheme
#def LHS_T(A,B,C):
#    H = A*Tnp1[1:-1,:] + B*Tnp1[1:-1,:] + C*Tnp1[1:-1,:]


def LHS_T(D,Tm0,Tm1):
    D2 = D*D
    tD2 = np.transpose(D2)

    ## Boundaries
    ## D2 and boundaries
    D2 = D*D
    tD2 = np.transpose(D2)

    D200 = D2[0,:]
    D20m = D2[-1,:]

    D2m0 = D2[:,0]
    D2mm = D2[:,-1]

    D2im = D2[:,-1]
    D2i0 = D2[:,0]

    D2ik = D2[:,1:-1]

    ## D2 transpose
    tD2kj = tD2[1:-1,:]

    ## D and boundaries
    D00 = D[0,0]
    D0m = D[0,-1]
    Dm0 = D[-1,0]
    Dmm = D[-1,-1]

    Dmk = D[-1,1:-1]
    D0k = D[0,1:-1]

    ## Helmoltz constants
    A = D2ik + (D2i0 *(Dm0*D0k - D00*Dmk)/(Dm0*D0m - D00*Dmm)) + (D2im*(Dmm*D0k - D0m*Dmk)/(D00*D0m - Dm0*D0m))
    B = tD2kj
    C = 3*(1/(2*cont.h))

    return RHST
