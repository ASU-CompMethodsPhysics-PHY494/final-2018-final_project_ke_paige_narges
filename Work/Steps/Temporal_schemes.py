## simulations
import numpy as np
import constants as cont

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

    RHST = NL(U1,V1,T1,D) - NL(U0,V0,T0,D) + ((4*T1 + T0)/2*cont.h)
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
    A = D2ik + (D2i0 *(Dm0*D0k - D00*Dmk)/(Dm0*D0m - D00*Dmm)) + \
       (D2im*(Dmm*D0k - D0m*Dmk)/(D00*D0m - Dm0*D0m))
    B = tD2kj
    C = 3*(1/(2*cont.h))

    return A,B,C


## right hand side of Temportal Scheme
def RHS_P(U1,V1,T1,U0,V0,T0,D,Tm0,Tmm,T2):
    D2 = D*D
    tD2 = np.transpose(D2)

    D2im = D2[:,-1]
    D2i0 = D2[:,0]

    tD20j = tD2[0,:]
    tD2mj = tD2[-1,:]

    RHSP = cont.asp*D*(-2*NL(U1,V1,U1,D) + NL(U0,V0,U0,D)) + \
           cont.asp*((-2*NL(U1,V1,V1,D) + NL(U0,V0,V0,D) + \
           cont.Gr*T2)*np.transpose(D)
    RHSP -= cont.asp**2*(D2i0 + D2im + tD20j + tD2mj)

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
    A = cont.aps**2*( D2ik - D2im * ((Dm0*D0k - D00*Dmk)/(Dm0*D0m - D00*Dmm)) -\
        D2i0 * ((Dmm*D0k - D0m*Dmk)/(D00*Dmm - Dm0*D0m))  )
    B = cont.aps**2*( tD2kj - ((tDmm*tDk0 - tDm0*tDkm)/(tD00*tDmm - tD0m*tDm0)) * tD20j \
        - ((tD0m*tDk0 - tD00*tDkm)/(tD0m*tDm0 - tD00*tDmm))*tD2mj  )

    return A,B

## right hand side of Temportal Scheme
def RHS_U(U1,V1,T1,U0,V0,T0,D,Tm0,Tmm,T2,P):
    D2 = D*D
    tD2 = np.transpose(D2)

    D2im = D2[:,-1]
    D2i0 = D2[:,0]

    tD20j = tD2[0,:]
    tD2mj = tD2[-1,:]

    RHSU = cont.asp*D*P + 2*NL(U1,V1,U1,D) - NL(U0,V0,U0,D) + ((4*U1 - U0)/2*cont.h) - cont.Gr*T2
    RHSU -= cont.asp**2*(D2i0 + D2im + tD20j + tD2mj)

    RHSV = cont.asp*P*np.transpose(D) + NL(U1,V1,V1,D) - NL(U0,V0,V0,D) + ((4*V1 + V0)/2*cont.h) - cont.Gr*T2
    RHSV -= cont.asp**2*(D2i0 + D2im + tD20j + tD2mj)

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
    C = 3*(1/(2*cont.h))

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
    A = ( D2ik - D2im * ((Dm0*D0k - D00*Dmk)/(Dm0*D0m - D00*Dmm)) \
        - D2i0 * ((Dmm*D0k - D0m*Dmk)/(D00*Dmm - Dm0*D0m))  )
    B = ( tD2kj - ((tDmm*tDk0 - tDm0*tDkm)/(tD00*tDmm - tD0m*tDm0)) * tD20j\
        - ((tD0m*tDk0 - tD00*tDkm)/(tD0m*tDm0 - tD00*tDmm))*tD2mj  )

    return A,B,C


def PUV(D, P, U, V, Z):
    tD = np.transpose(D)
    P2 = P + (3/(2*h)) * Z
    U2 = U - cont.asp * D * Z
    V2 = V  - cont.asp * Z * tD

    return P2, U2, V2

def boundary_values(M,N,Re)
  boudary_u = np.zeros([M+1,N+1])
  boudary_u[:,0] = Re
  boudary_v = 0.0
  boudary_T = np.zeros([M+1,M+1])
  boudary_T[:,0] = 0.5
  boudary_T[:,N] =-0.5

def eigens_P(M,N,asp,D,D2,TD,TD2):
  eign_vec_P1  = np.array([M-1,M-1])
  eigen_val_P1 = np.array([M-1])
  eign_vec_P2  = np.array([N-1,N-1])
  eigen_val_P2 = np.array([N-1])
  tempx = np.array([M-1,M-1])
  tempy = np.array([N-1,N-1])
  
  D00_MM_DM0_0M   = D[0,0]*D[M,M]   - D[M,0]*D[0,M]
  TD00_NN_TD0N_N0 = TD[0,0]*TD[N,N] - TD[0,N]*TD[N,N] 
  for i in range(M+1): # generate the coefficient matrix column-wise
    tempx[i,1:M-1] =-D2[i,0] * (D[M,M]*D[0,1:M-1]-D[0,M]*D[M,1:M-1])/D00_MM_DM0_0M \
                    +D2[i,M] * (D[M,0]*D[0,1:M-1]-D[0,0]*D[M,1:M-1])/D00_MM_DM0_0M 
  tempx = asp**2 * (D2[1:M-1,1:N-1] - tempx)
  for k in range(M+1): # generate the coefficient matrix row-wise
    tempy[1:M-1,j] = -TD2[0,j] * (TD[N,N]*DT[1:M-1,0] - TD[N,0]*TD[1:M-1,N])/TD00_NN_TD0N_N0 \
                     +TD2[N,j] * (TD[0,N]*TD[1:M-1,0] - TD[0,0]*TD[1:M-1,N])/TD00_NN_TD0N_N0
  tempy = 4.0 * (TD2[1:M-1,1:M-1] - tempy)
  eigen_vector1, eigen_value1 = np.linalg.eig(tempx)
  eigen_vector2, eigen_value2 = np.linalg.eig(tempy)

  return eigen_vec_P1, eigen_val_P1, eigen_vec_P2, eigen_val_P2

def eigens_T(M,N,asp,D,D2,TD,TD2):
  eign_vec_T1  = np.array([M-1,M-1])
  eigen_val_T1 = np.array([M-1])

  eign_vec_T2  = np.array([N-1,N-1])
  eigen_val_T2 = np.array([N-1])
  
  tempx = np.array([M-1,M-1])
  tempy = np.array([N-1,N-1])
  
  D00_MM_DM0_0M   = D[0,0]*D[M,M]   - D[M,0]*D[0,M]
  TD00_NN_TD0N_N0 = TD[0,0]*TD[N,N] - TD[0,N]*TD[N,N] 
  for i in range(M+1): # Neumann boundary conditions
    tempx[i,1:M-1] =-D2[i,0] * (D[M,M]*D[0,1:M-1]-D[0,M]*D[M,1:M-1])/D00_MM_DM0_0M \
                    +D2[i,M] * (D[M,0]*D[0,1:M-1]-D[0,0]*D[M,1:M-1])/D00_MM_DM0_0M 
  tempx = asp**2 * (D2[1:M-1,1:N-1] - tempx)
                       # Dirichlet boundary conditions 
  tempy = 4.0 * TD2[1:N-1,1:N-1] 

  eigen_vec_T1, eigen_val_T1 = np.linalg.eig(tempx)
  eigen_vec_T2, eigen_val_T2 = np.linalg.eig(tempy)

  return eigen_vec_T1, eigen_val_T1, eigen_vec_T2, eigen_val_T2

def eigens_vel(M,N,asp,D,D2,TD,TD2):
  eign_vec_vel1  = np.array([M-1,M-1])
  eigen_val_vel1 = np.array([M-1])
  eign_vec_vel2  = np.array([N-1,N-1])
  eigen_val_vle2 = np.array([N-1])
  tempx = np.array([M-1,M-1])
  tempy = np.array([N-1,N-1])
  
  D00_MM_DM0_0M   = D[0,0]*D[M,M]   - D[M,0]*D[0,M]
  TD00_NN_TD0N_N0 = TD[0,0]*TD[N,N] - TD[0,N]*TD[N,N] 
  # Dirichelt bounday condition for both directions
  tempx = asp**2 * D2[1:M-1,1:N-1] 
  tempy = 4.0   * TD2[1:N-1,1:N-1] 
  eigen_vec_vel1, eigen_val_vel1 = np.linalg.eig(tempx)
  eigen_vec_vel2, eigen_val_vel2 = np.linalg.eig(tempy)

  return eigen_vec_vel1, eigen_val_vel1, eigen_vec_vel2, eigen_val_vel2

def eigens_phi(M,N,asp,D,D2,TD,TD2):
  eign_vec_phi1  = np.array([M-1,M-1])
  eigen_val_phi1 = np.array([M-1])
  eign_vec_phi2  = np.array([N-1,N-1])
  eigen_val_phi2 = np.array([N-1])
  tempx = np.array([M-1,M-1])
  tempy = np.array([N-1,N-1])
  
  D00_MM_DM0_0M   = D[0,0]*D[M,M]   - D[M,0]*D[0,M]
  TD00_NN_TD0N_N0 = TD[0,0]*TD[N,N] - TD[0,N]*TD[N,N] 
  for i in range(M+1): # generate the coefficient matrix column-wise
    tempx[i,1:M-1] =-D2[i,0] * (D[M,M]*D[0,1:M-1]-D[0,M]*D[M,1:M-1])/D00_MM_DM0_0M \
                    +D2[i,M] * (D[M,0]*D[0,1:M-1]-D[0,0]*D[M,1:M-1])/D00_MM_DM0_0M 
  tempx = asp**2 * (D2[1:M-1,1:N-1] - tempx)
  for k in range(M+1): # generate the coefficient matrix row-wise
    tempy[1:M-1,j] = -TD2[0,j] * (TD[N,N]*DT[1:M-1,0] - TD[N,0]*TD[1:M-1,N])/TD00_NN_TD0N_N0 \
                     +TD2[N,j] * (TD[0,N]*TD[1:M-1,0] - TD[0,0]*TD[1:M-1,N])/TD00_NN_TD0N_N0
  tempy = 4.0 * (TD2[1:M-1,1:M-1] - tempy)
  eigen_vec_phi1, eigen_val_phi1 = np.linalg.eig(tempx)
  eigen_vec_phi2, eigen_val_phi2 = np.linalg.eig(tempy)

  return eigen_vec_phi1, eigen_val_phi1, eigen_vec_phi2, eigen_val_phi2

def Helmholtz_sovler(M,N,sigma,eig_val1,eig_vec1,eig_val2,eig_vec2,H):
  tol = 1e-10
  tempH = np.linalg.solve(eig_vec1,H)
  tempH = np.matmul(tempH,eig_vec2)

  for j in range(N):
    for i in range(M):
      if abs(eig_val1(i))<tol and abs(eig_val2(j))<tol and abs(sigma)<tol:
        X[i,j] = 0.0
      else:
        X[i,j] = tempH[i,j]/(eig_val1[i]+eig_val2[j]+sigma)

  tempX = np.linalg.solve(np.transpose(eig_vec2,np.transpose(X)))
  tempX = np.transpose(tempX)
  X = np.matmul(eig_vec1,tempX)
  
  return X
         
          











