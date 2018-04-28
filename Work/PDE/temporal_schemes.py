## simulations
import numpy as np
import input_parameters as cont

## NLT
def NL(U,V,X,D):
    NLX = cont.asp* U * (D*X) + cont.asp* V * (X*np.transpose(D))
    return NLX

## right hand side of Temportal Scheme
def RHS_T(U1,V1,T1,U0,V0,T0,D):
    ## T boundary conditions
    Tm0 = T1[:,0]
    Tmm = T1[:,-1]

    D2 = D*D
    tD2 = np.transpose(D2)
    tD2mj = tD2[-1:]
    tD20j = tD2[0,:]

    RHST = NL(U1,V1,T1,D) - NL(U0,V0,T0,D) + ((4*T1 + T0)/2*cont.dt)
    RHST -= (Tm0*tD20j  + Tmm*tD2mj)

    return RHST

## left hand side of Temporal Scheme
#def LHS_T(A,B,C):
#    H = A*Tnp1[1:-1,:] + B*Tnp1[1:-1,:] + C*Tnp1[1:-1,:]


def LHS_T(D):
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
    C = 3*(1/(2*cont.dt))

    return A,B,C


## right hand side of Temportal Scheme
def RHS_P(U1,V1,T1,U0,V0,T0,D,Tm0,Tmm,T2):
    D2 = D*D
    tD = np.transpose(D)
    tD2 = np.transpose(D2)

    ## D Boundaries
    Dmm = D[-1,-1]
    D0m = D[0,-1]
    D00 = D[0,0]
    Dm0 = D[-1,0]

    tDmm = tD[-1,-1]
    tD0m = tD[0,-1]
    tD00 = tD[0,0]
    tDm0 = tD[-1,0]

    ## D2 boundaries
    D2im = D2[:,-1]
    D2i0 = D2[:,0]

    tD20j = tD2[0,:]
    tD2mj = tD2[-1,:]


    RHSP1 = cont.asp*D*(-2*NL(U1,V1,U1,D) + NL(U0,V0,U0,D))
    RHSP2 = cont.asp*((-2*NL(U1,V1,V1,D) + NL(U0,V0,V0,D) + cont.Ri*cont.Re*cont.Re*T2)*np.transpose(D)

    rhs10j = RHSP1[0,0:m]
    rhs1mj = RHSP1[-1,0:m]
    rhs2i0 = RHSP2[0:m,0]
    rhs2im = RHSP2[0:m,-1]


    RHSP = RHSP1 + RHSP2 - cont.asp*(D2i0*((rhs10j*Dmm - rhs1mj*D0m)/(D00*Dmm - D0m*Dm0)) + D2im*((rhs10j*Dm0 - rhs1mj*D00)/(Dm0*D0m - D00*Dmm)) + tD20j*((rhs20j*tDmm - rhs2mj*tD0m)/(tD00*tDmm - tD0m*tDm0)) + tD2mj*(rhs20j*tDm0 - rhs2mj*tD00)/(tDm0*tD0m - tD00*tDmm)))

    return RHSP


def LHS_P(D):
    tD = np.transpose(D)
    D2 = D*D
    tD2 = np.transpose(D2)

    ## D2
    D2ik = D2[:,1:-1]
    D2i0 = D2[:,0]
    D2im = D2[:,-1]

    ## D2 transpose
    tD2kj = tD2[1:-1,:]
    tD20j = tD2[0,:]
    tD2mj = tD2[-1,:]

    ## D and boundaries
    D00 = D[0,0]
    D0m = D[0,-1]
    Dm0 = D[-1,0]
    Dmm = D[-1,-1]

    Dmk = D[-1,1:-1]
    D0k = D[0,1:-1]

    ## D transpose
    tD00 = tD[0,0]
    tD0m = tD[0,-1]
    tDm0 = tD[-1,0]
    tDmm = tD[-1,-1]

    tDkm = tD[-1,1:-1]
    tDk0 = tD[0,1:-1]

    ## Helmoltz constants
    A = cont.aps**2*( D2ik - D2im * ((Dm0*D0k - D00*Dmk)/(Dm0*D0m - D00*Dmm)) - D2i0 * ((Dmm*D0k - D0m*Dmk)/(D00*Dmm - Dm0*D0m))  )
    B = cont.aps**2*( tD2kj - ((tDmm*tDk0 - tDm0*tDkm)/(tD00*tDmm - tD0m*tDm0)) * tD20j - ((tD0m*tDk0 - tD00*tDkm)/(tD0m*tDm0 - tD00*tDmm))*tD2mj  )

    return A,B

## right hand side of Temportal Scheme
def RHS_U(U1,V1,T1,U0,V0,T0,D,Tm0,Tmm,T2,P):
    D2 = D*D
    tD2 = np.transpose(D2)

    D2im = D2[:,-1]
    D2i0 = D2[:,0]

    tD20j = tD2[0,:]
    tD2mj = tD2[-1,:]

    r,s = np.shape(U0)

    ## U orginial BC
    WU0j = cont.Re * np.asmatrix(np.ones((1,r)))
    WUmj = np.asmatrix(np.zeros((1,r)))
    WUi0  = np.transpose(WUmj)
    WUim =  WUi0

    ## V original BC
    WV0j = WUmj
    WVmj = WUmj
    WVi0 = WUi0
    WVim = WVi0


    RHSU = cont.asp*D*P + 2*NL(U1,V1,U1,D) - NL(U0,V0,U0,D) + ((4*U1 - U0)/2*cont.dt
    ) - cont.Ri*cont.Re*cont.Re*T2
    RHSU -= cont.asp**2*(D2i0*WU0j + D2im*WUmj + WUi0*tD20j + WUim*tD2mj)

    RHSV = cont.asp*P*np.transpose(D) + NL(U1,V1,V1,D) - NL(U0,V0,V0,D) + ((4*V1 + V0)/2*cont.dt) - cont.Ri*cont.Re*cont.Re*T2
    RHSV -= cont.asp**2*(D2i0*WV0j + D2im*WVmj + WVi0*tD20j + WVim*tD2mj)

    return RHSU, RHSV


def LHS_U(D):
    D2 = D*D
    tD2 = np.transpose(D2)

    ## Boundaries
    ## D2 and boundaries
    D2 = D*D
    tD2 = np.transpose(D2)

    ## D2 transpose
    D2ik = D2[:,1:-1]
    tD2kj = tD2[1:-1,:]

    ## Helmoltz constants
    A = D2ik
    B = tD2kj
    C = 3*(1/(2*cont.dt))

    return A,B,C


## right hand side of Temportal Scheme
def RHS_Z(D, U, Vs):
    tD = np.transpose(D)

    RHSZ = cont.asp*(D*T + V*tD)
    RHSZ *= 0.25

    return RHSZ

def LHS_Z(D):
    tD = np.transpose(D)
    D2 = D*D
    tD2 = np.transpose(D2)

    ## D2
    D2ik = D2[:,1:-1]
    D2i0 = D2[:,0]
    D2im = D2[:,-1]

    ## D2 transpose
    tD2kj = tD2[1:-1,:]
    tD20j = tD2[0,:]
    tD2mj = tD2[-1,:]

    ## D and boundaries
    D00 = D[0,0]
    D0m = D[0,-1]
    Dm0 = D[-1,0]
    Dmm = D[-1,-1]

    Dmk = D[-1,1:-1]
    D0k = D[0,1:-1]

    ## D transpose
    tD00 = tD[0,0]
    tD0m = tD[0,-1]
    tDm0 = tD[-1,0]
    tDmm = tD[-1,-1]

    tDkm = tD[-1,1:-1]
    tDk0 = tD[0,1:-1]

    ## Helmoltz constants
    A = ( D2ik - D2im * ((Dm0*D0k - D00*Dmk)/(Dm0*D0m - D00*Dmm)) - D2i0 * ((Dmm*D0k - D0m*Dmk)/(D00*Dmm - Dm0*D0m))  )
    B = ( tD2kj - ((tDmm*tDk0 - tDm0*tDkm)/(tD00*tDmm - tD0m*tDm0)) * tD20j - ((tD0m*tDk0 - tD00*tDkm)/(tD0m*tDm0 - tD00*tDmm))*tD2mj  )

    return A,B,C


def PUV(D, P, U, V, Z):
    tD = np.transpose(D)
    P2 = P + (3/(2*dt)) * Z
    U2 = U - cont.asp * D * Z
    V2 = V  - cont.asp * Z * tD

    return P2, U2, V2
