## simulations
import numpy as np
import constants as cont

# NLT : nonlinear term of temperature equation
def NLT(M,N,asp,U,V,T,D,TD):
	X = np.zeros([M+1,N+1])
	X = asp*U*np.matmul(D,T) + 2*V*np.matmul(T,TD)
# NLX = cont.asp* U * (D*X) + cont.asp* V * (X*np.transpose(D))
        return X

# RHS_T: right hand side of temperature equation
def RHS_T(NLT,NLT_old,D,TD):
	## T boundary conditions
	#    Tm0 = T1[:,0]
	#    Tmm = T1[:,-1]
	#    D2 = D*D
	#    tD2 = np.transpose(D2)
	#    tD2mj = tD2[-1:]
	#    tD20j = tD2[0,:]
   RHST = ( 2.0*NLT - NLT_old + (4.0*T-Told) )/(2.0*cont.dt)
   RHST = RHST*cont.Pr
   for i in range(M-1): # only update the interior points
      RHST[i+1,1:N-1] = RHS[i+1,1:N-1]-4.0*Tnew[i+1,0]*TD2[0,1:N-1] \
			 -4.0*Tnew[i+1,N]*TD2[N,1:M-1]
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
    u_top = np.zeros([M+1,1])
    T_top = np.zeros([M+1,1])
    T_bot = np.zeros([M+1,1])
    u_top = Re
    T_top[:,0] = 0.5
    T_bot[:,N] =-0.5

def eigens_P(M,N,asp,D,D2,TD,TD2):
    eigen_vec_P1  = np.array([M-1,M-1])
    eigen_val_P1 = np.array([M-1])
    eigen_vec_P2  = np.array([N-1,N-1])
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
    eigen_vec_vel1  = np.array([M-1,M-1])
    eigen_val_vel1 = np.array([M-1])
    eigen_vec_vel2  = np.array([N-1,N-1])
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
    eigen_vec_phi1  = np.array([M-1,M-1])
    eigen_val_phi1 = np.array([M-1])
    eigen_vec_phi2  = np.array([N-1,N-1])
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

def Helmholtz_solver(M,N,sigma,eig_val1,eig_vec1,eig_val2,eig_vec2,H):
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

def T(M,N,Pr,dt,TD2,NLT_old,NLT,Told,T,eig_vecT1,eig_valT1,eig_vecT2,eig_valT2):
    Tnew = np.zeros([M+1,N+1])
    rhs  = np.zeros([M+1,N+1])
    rhs  = Pr*( 2.0*NLT - NLT_old - (4.0*T-Told)/(2.0*dt) )
    for j in range(1,M):
        for i in range(1,N):  # the top and bot boundaries of T are given
            rhs[i,j] = rhs[i,j] - 4.0*Tnew[i,0]*TD2[0,j] - 4.0*Tnew[i,N]*TD2[N,j]
    sigma= 3.0*Pr/(2.0*dt)
    Tnew[1:M,1:N]=Helmholtz_solver(M,N,sigma,eig_valT1,eig_vecT1,eig_valT2,eig_vecT2,rhs[1:M,1:N])
    for j in range(1,M)      # c1 and c2 need to be defined
        Tnew[0,j] = np.dot(D[M,0]*D[0,1:M]-D[0,0]*D[M,1:M],Tnew[1:M,j])/c1
        Tnew[-1,j]= np.dot(D[M,M]*D[0,1:M]-D[0,M]*D[M,1:M],Tnew[1:M,j])/c2
  # update top and bottom boundaries                                   
    return Tnew

def Pressure(M,N,dt,Gr,u,uold,v,vold,Tnew,D,D2,TD,TD2,eig_valP1,eig_vecP1,eig_valP2,eig_vecP2,\
             NLU,NLUold,NLV,NLVold,LU,LV,LUold,LVold,U0,V0)
    RHS = np.zeros(M+1,N+1)
    rhs1= np.zeros(M+1,N+1)
    rhs2= np.zeros(M+1,M+1)
    RHS = asp*np.matmul(D,-2*NLU+NLUold,) + asp*np.matmul(-2*NLV+NLVold+Gr*Tnew,TD)
    rhs1[0,:] =(-3*u0[0,:]+4*u[0,:]-uold[0,:]) / (2.0*dt) \
                -2*NLU[0,:]+NLUold[0,:] +2*LU[0,:]-LUold[0,:]
    rhs1[-1,:]=(-3*u0[-1,:]+4*u[-1,:]-uold[-1,:]) / (2.0*dt) \
                -2*NLU[-1,:]+NLUold[-1,:] +2*LU[-1,:]-LUold[-1,:]
    rhs2[:,0] =(-3*v0[:,0]+4*v[:,0]-vold[:,0]) / (2.0*dt) \
                -2*NLV[0,:] +NLVold[:,0]  +2*LV[:,0]-LVold[:,0] + Gr*Tnew[:,0]
    rhs2[:,-1]=(-3*v0[:,-1] +4*v[:,-1]-uold[:,-1]) / (2.0*dt) \
                -2*NLV[:,-1]+NLVold[:,-1] +2*LU[:,-1]-LVold[:,-1] + Gr*Tnew[:,-1]
    for j in range(1,M):  # c1,c2,c3,c4 need to be defined
        RHS[:,j] = RHS[:,j]-asp*D2[:,0] *(D(M,N)*rhs1[0,j] -D[0,M]*rhs1[M,j] ) / c2 \
                          -asp*D2[:,M]  *(D[M,0]*rhs1[0,j] -D[0,0]*rhs1[N,j] ) / c1 \
                          -2*TD2[0,j]  *(TD[M,M]*rhs2[:,0] -TD[M,0]*rhs2[:,M] )/ c4 \
                          -2*TD2[M,j]  *(TD[0,M]*rhs2[:,0] -TD[0,0]*rhs2[:,M] )/ c3
    sigma=0.0
    P=0.0
    P[1:M,1:N]=Helmholtz_solver(M,N,sigma,eig_valP1,eig_vecP1,eig_valP2,eig_vecP2,RHS[1:M,1:N])
    for j in range(0,M+1): 
        p[M,j]=((D[M,0]*rhs1[0,j]- D[0,0]*rhs1[M,j]) /
                 -np.dot(D[N,0]*D[0,1:N]-D[0,0]*D[N,1:N], P[1:M,j])) / c1
     
        p[0,j]=((D(M,M)*rhs1[0,j]- D[0,M]*rhs1(M,j)) /
                 -np.dot(D(M,M)*D[0,1:N]-D[0,M]*D(M,1:M), P[1:M,j])) / c2

    for i in range(0,M+1):
        P[i,M]=((TD[0,M]*rhs2[i,0]-TD[0,0]*rhs2[i,M]) / 
                 -np.dot(TD[0,M]*TD[1:M,0]-TD[0,0]*TD[1:M,M], P[i,1:M])/c3

        P[i,0]=((TD[M,M]*rhs2[i,0]-TD[M,0]*rhs2[i,M]) /
                 -np.dot(TD[M,M]*TD[1:M,0]-TD[M,0]*TD[1:M,M], P[i,1:M]))/c4                     
    return p

def Velocity(M,N,dt,Gr,P,u,uold,v,vold,Tnew,D,D2,TD,TD2,eig_valv1,eig_vecv1,eig_valv2,eig_vecv2,\
             NLU,NLUold,NLV,NLVold,U0,V0)
                
    rhs1 =(asp*np.matmul(D,P) +2*NLU -NLU_old -(4*U-Uold)/(2*dt))
    rhs2 =(2*np.matmul(P,TD)  +2*NLV -NLV_old -(4*V-Vold)/(2*dt)) - Gr*Tnew)

    for j in range(1,N):
        rhs1[:,j]=rhs1[:,j] -D2[:,0] *u0[0,j] -D2[:,N]*u0[N,j] /
                            -u0[:,0] *TD2[0,j]-u0[:,M]*TD2[M,j]
        rhs2[:,j]=rhs2[:,j] -D2[:,0] *v0[0,j] -D2[:,N]*v0[N,j] /
                            -v0[:,0] *TD2[0,j]-v0[:,M]*TD2[M,j]
    sigma = 3.0/(2*dt)
    Unew[1:M,1:N]=Helmholtz_solver(M,N,sigma,eig_valv1,eig_vecv1,eig_valv2,eig_vecv2,rhs1[1:M,1:N])
    Vnew[1:M,1:N]=Helmholtz_solver(M,N,sigma,eig_valv1,eig_vecv1,eig_valv2,eig_vecv2,rhs2[1:M,1:N])
# impose exact boundary velocity    
    Unew(:,0) = u0(:,0);  Vnew(:,0) = v0(:,0)
    Unew(:,M) = u0(:,M);  Vnew(:,M) = v0(:,N)
    Unew(0,:) = u0(0,:);  Vnew(0,:) = v0(0,:)
    Unew(N,:) = u0(N,:);  Vnew(N,:) = v0(N,:)
    return Unew, Vnew

def correction(M,N,Unew,Vnew,P,D,TD,D2,TD2,eig_val_phi1,eig_vec_phi1,eig_val_phi2,eig_vec_phi2)
    RHS=np.zeros(M+1,N+1)
    RHS=np.matmul(D,Unew) +np.matmul(Vnew,TD)
    phi=np.zeros(M+1,N+1)
    sigma=0.0
    phi[1:M,1:N]=Helmholtz_solver(M,N,sigma,eig_val_phi1,eig_vec_phi1,eig_val_phi2,eig_vec_phi2,RHS[1:M,1:N])
    # define c1, c2, c3, c4
    for j in range(1,M):  # update the boundary of phi
        phi(N,j] = -np.dot(D[M,0]*D[0,1:M]-D[0,0]*D[M,1:M],phi[1:M,j]) /c1
        phi[0,j] = -np.dot(D(N,N)*D[0,1:N]-D[0,N]*D[N,1:N],phi[1:M,j]) /c2

    for j in range(1,M):
        phi(j,M) = -np.dot(phi[j,1:M], TD[0,M]*TD[1:M,0]-TD[0,0]*TD[1:M,M]) /c3
        phi(j,0) = -np.dot(phi[j,1:M], TD[M,M]*TD[1:M,0]-TD[M,0]*TD[1:M,M]) /c4

    P    = P + 3/(4*dt)*phi
    Unew = Unew - np.matmul(D,phi)
    Vnew = Vnew - np.matmul(phi,TD)
    
    return Unew, Vnew, P
            

                     

    
    
                     
                     
    

    
    
             
          











