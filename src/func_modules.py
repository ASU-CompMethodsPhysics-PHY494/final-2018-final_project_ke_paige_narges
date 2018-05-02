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

## Exact Boundary Conditions    
Uexact = np.asmatrix(diff.U_0(x))
Vexact = np.asmatrix(np.zeros([cont.M+1,cont.N+1]))  

## Commonly used denominator
c1 = D[0,0]*D[-1,-1] - D[-1,0]*D[0,-1]
c2 = -c1
c3 = TD[0,0]*D[-1,-1] - TD[0,-1]*TD[-1,0]
c4 = -c3

## NLT
def NL(U,V,X):
    NLX = np.asmatrix(np.asarray(U) * np.asarray(D*X)) +\
          np.asmatrix(np.asarray(V) * np.asarray(X*TD))
    NLX  = NLX * 2
    return NLX

## LU
def LU(U,V):
    LU = (U*TD2) - (D*V)*TD
    LU = LU * 4
    return LU

## LV
def LV(U,V):
    LV = (D2*V) - (D)*(U*TD)
    LV = LV * 4
    return LV


'''Eigenvalues and Eigenvectors'''


## Temperature
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
    tempx[1:cont.M,1:cont.M] = cont.asp**2 * (D2[1:cont.M,1:cont.M] + tempx[1:cont.M,1:cont.M])
                       # Dirichlet boundary conditions 
   
    tempy[1:cont.N,1:cont.N] = cont.asp**2 * TD2[1:cont.N,1:cont.N] 
    
    eigen_val_T1[1:cont.M], eigen_vec_T1[1:cont.M,1:cont.M] = np.linalg.eig(tempx[1:cont.M,1:cont.M])
    eigen_val_T2[1:cont.N]  , eigen_vec_T2[1:cont.N,1:cont.N] = np.linalg.eig(tempy[1:cont.N,1:cont.N] )

    return eigen_vec_T1, eigen_val_T1, eigen_vec_T2, eigen_val_T2


'''RHS for T, P, U, V, Phi'''


## Temperature
def RHS_T(U0,V0,T0,U1,V1,T1):
    
    RHST = cont.Pr*(2*NL(U1,V1,T1) - NL(U0,V0,T0) - ( (4*T1 - T0)/(2*cont.dt) ) )
    
    for j in range(1,cont.M):
        for i in range(1,cont.N):
            RHST[i,j] = RHST[i,j] - cont.asp*cont.asp*(T1[i,0]*TD2[0,j] + T1[i,-1]*TD2[-1,j]) 
            ##RHST[i,j] = RHST[i,j] - (T1[i,0]*TD2[0,j] + T1[i,-1]*TD2[-1,j]) 
        
    return RHST

## Pressure
def RHS_P(U0,V0,T0,U1,V1,T1,T2):
    
    RHS = np.asmatrix(np.zeros([cont.M+1,cont.N+1]))
    rhs1= np.asmatrix(np.zeros([cont.M+1,cont.N+1]))
    rhs2= np.asmatrix(np.zeros([cont.M+1,cont.N+1]))

    
    RHS = cont.asp*D*( -2*NL(U1,V1,U1) + NL(U0,V0,U0) )\
        + cont.asp*( -2*NL(U1,V1,V1) + NL(U0,V0,V0) + cont.Ri*cont.Re*cont.Re*T2)*TD 
    rhs1[0,:] =(-3*Uexact[0,:]+4*U1[0,:]-U0[0,:]) / (2.0*cont.dt) \
              + (-2*NL(U1,V1,U1))[0,:] + NL(U0,V0,U0)[0,:] +2*LU(U1,V1)[0,:]-LU(U0,V0)[0,:]
    rhs1[-1,:]=(-3*Uexact[-1,:]+4*U1[-1,:]-U0[-1,:]) / (2.0*cont.dt) \
              + (-2*NL(U1,V1,U1))[-1,:] + NL(U0,V0,U0)[-1,:] +2*LU(U1,V1)[-1,:]-LU(U0,V0)[-1,:]
    rhs2[:,0] =(-3*Vexact[:,0]+4*V1[:,0]-V0[:,0]) / (2.0*cont.dt) \
              + (-2*NL(U1,V1,V1))[:,0] + NL(U0,V0,V0)[:,0] +2*LV(U1,V1)[:,0]\
              -LV(U0,V0)[:,0]+ cont.Ri*cont.Re*cont.Re*T2[:,0]
    rhs2[:,-1]=(-3*Vexact[:,-1]+4*V1[:,-1]-V0[:,-1]) / (2.0*cont.dt) \
              + (-2*NL(U1,V1,V1))[:,-1] + NL(U0,V0,V0)[:,-1] +2*LV(U1,V1)[:,-1]\
              -LV(U0,V0)[:,-1] + cont.Ri*cont.Re*cont.Re*T2[:,-1]
            
    for j in range(1,cont.M):  # c1,c2,c3,c4 need to be defined
        RHS[:,j] = RHS[:,j]-cont.asp*D2[:,0] *(D[cont.M,cont.N]*rhs1[0,j] -D[0,cont.M]*rhs1[cont.M,j] ) / c1 \
                          -cont.asp*D2[:,cont.M]  *(D[cont.M,0]*rhs1[0,j] -D[0,0]*rhs1[cont.N,j] ) / c2 \
                          -2*TD2[0,j]  *(TD[cont.M,cont.M]*rhs2[:,0] -TD[cont.M,0]*rhs2[:,cont.M] )/ c3 \
                          -2*TD2[cont.M,j]  *(TD[0,cont.M]*rhs2[:,0] -TD[0,0]*rhs2[:,cont.M] )/ c4
    RHS = np.asarray(RHS)
    return rhs1, rhs2, RHS

## Velocity
def RHS_U(U0,V0,T0,U1,V1,T1,T2,P):

    RHSU = cont.asp*D*P + 2*NL(U1,V1,U1)  - NL(U0,V0,U0) - (4*U1 - U0)/(2*cont.dt)
    RHSV = cont.asp*P*TD + 2*NL(U1,V1,V1) - NL(U0,V0,V0) - (4*V1 - V0)/(2*cont.dt) - cont.Ri*cont.Re*cont.Re*T2
    
    for j in range(1,cont.N):
        RHSU[:,j]= RHSU[:,j] - cont.asp*cont.asp*D2[:,0] *Uexact[0,j]   - cont.asp*cont.asp*D2[:,cont.N]*Uexact[cont.N,j] \
                             - cont.asp*cont.asp*Uexact[:,0] * TD2[0,j] - cont.asp*cont.asp*Uexact[:,cont.M]*TD2[cont.M,j]
        RHSV[:,j]= RHSV[:,j] - cont.asp*cont.asp*D2[:,0] *Vexact[0,j]   - cont.asp*cont.asp*D2[:,cont.N]*Vexact[cont.N,j] \
                             - cont.asp*cont.asp*Vexact[:,0] *TD2[0,j]  - cont.asp*cont.asp*Vexact[:,cont.M]*TD2[cont.M,j]
    return RHSU, RHSV


## Phi
def RHS_phi(U,V):
    
    RHS = cont.asp*(D*U + V*TD)

    return RHS

## Pressure
def eigens_P():
    eigen_vec_P1  = np.zeros([cont.M+1,cont.M+1])
    eigen_val_P1 = np.zeros([cont.M+1])
    eigen_vec_P2  = np.zeros([cont.N+1,cont.N+1])
    eigen_val_P2 = np.zeros([cont.N+1])
    tempx = np.zeros([cont.M+1,cont.M+1])
    tempy = np.zeros([cont.N+1,cont.N+1])
  
    D00_MM_DM0_0M   = D[0,0]*D[cont.M,cont.M]   - D[cont.M,0]*D[0,cont.M]
    TD00_NN_TD0N_N0 = TD[0,0]*TD[cont.N,cont.N] - TD[0,cont.N]*TD[cont.N,0]
    
    for i in range(cont.M): # generate the coefficient matrix column-wise
        tempx[i,1:cont.M] = -D2[i,0] * (D[cont.M,cont.M]*D[0,1:cont.M]-D[0,cont.M]*D[cont.M,1:cont.M])/D00_MM_DM0_0M \
                          + D2[i,cont.M] * (D[cont.M,0]*D[0,1:cont.M]-D[0,0]*D[cont.M,1:cont.M])/D00_MM_DM0_0M 
    ##tempx[1:cont.M,1:cont.N] = (D2[1:cont.M,1:cont.N] + tempx[1:cont.M,1:cont.N])
    tempx[1:cont.M,1:cont.N] = cont.asp**2*(D2[1:cont.M,1:cont.N] + tempx[1:cont.M,1:cont.N])
    
    for j in range(cont.M): # generate the coefficient matrix column-wise
        tempy[1:cont.N,j] = np.asarray(np.transpose(-(TD2[0,j] * (TD[cont.N,cont.N]*TD[1:cont.N,0]-TD[cont.N,0]*TD[1:cont.N,cont.N])/TD00_NN_TD0N_N0) \
                          + (TD2[cont.N,j] * (TD[0,cont.N]*TD[1:cont.N,0]-TD[0,0]*TD[1:cont.N,cont.N])/TD00_NN_TD0N_N0)))
    ##tempy[1:cont.M,1:cont.N] = (TD2[1:cont.M,1:cont.N] + tempy[1:cont.M,1:cont.N])
    tempy[1:cont.M,1:cont.N] = cont.asp**2*(TD2[1:cont.M,1:cont.N] + tempy[1:cont.M,1:cont.N])
    
    eigen_val_P1[1:cont.M], eigen_vec_P1[1:cont.M,1:cont.M] = np.linalg.eig(tempx[1:cont.M,1:cont.M])
    eigen_val_P2[1:cont.N], eigen_vec_P2[1:cont.N,1:cont.N] = np.linalg.eig(tempy[1:cont.N,1:cont.N])
    
    return eigen_vec_P1, eigen_val_P1, eigen_vec_P2, eigen_val_P2

## Velocity
def eigens_vel():
    eigen_vec_vel1  = np.zeros([cont.M,cont.M+1])
    eigen_val_vel1 = np.zeros([cont.M+1])
    eigen_vec_vel2  = np.zeros([cont.N+1,cont.N+1])
    eigen_val_vel2 = np.zeros([cont.N+1])
    tempx = np.zeros([cont.M+1,cont.M+1])
    tempy = np.zeros([cont.N+1,cont.N+1])

    tempx[1:cont.M,1:cont.N]  = cont.asp*cont.asp*D2[1:cont.M,1:cont.N] 
    tempy[1:cont.N,1:cont.N]  =  cont.asp*cont.asp*TD2[1:cont.N,1:cont.N] 

    eigen_val_vel1[1:cont.M], eigen_vec_vel1[1:cont.M,1:cont.M] = np.linalg.eig(tempx[1:cont.M,1:cont.M])
    eigen_val_vel2[1:cont.N], eigen_vec_vel2[1:cont.N,1:cont.N] = np.linalg.eig(tempy[1:cont.N,1:cont.N] )

    return eigen_vec_vel1, eigen_val_vel1, eigen_vec_vel2, eigen_val_vel2

## Phi
def eigens_phi():
    eigen_vec_phi1  = np.zeros([cont.M+1,cont.M+1])
    eigen_val_phi1 = np.zeros([cont.M+1])
    eigen_vec_phi2  = np.zeros([cont.N+1,cont.N+1])
    eigen_val_phi2 = np.zeros([cont.N+1])
    tempx = np.zeros([cont.M+1,cont.M+1])
    tempy = np.zeros([cont.N+1,cont.N+1])
  
    D00_MM_DM0_0M   = D[0,0]*D[cont.M,cont.M]   - D[cont.M,0]*D[0,cont.M]
    TD00_NN_TD0N_N0 = TD[0,0]*TD[cont.N,cont.N] - TD[0,cont.N]*TD[cont.N,0]
    
    for i in range(cont.M): # generate the coefficient matrix column-wise
        tempx[i,1:cont.M] = -D2[i,0] * (D[cont.M,cont.M]*D[0,1:cont.M]-D[0,cont.M]*D[cont.M,1:cont.M])/D00_MM_DM0_0M \
                          + D2[i,cont.M] * (D[cont.M,0]*D[0,1:cont.M]-D[0,0]*D[cont.M,1:cont.M])/D00_MM_DM0_0M 
    ##tempx[1:cont.M,1:cont.N] = (D2[1:cont.M,1:cont.N] + tempx[1:cont.M,1:cont.N])
    tempx[1:cont.M,1:cont.N] = cont.asp**2*(D2[1:cont.M,1:cont.N] + tempx[1:cont.M,1:cont.N])
    
    for j in range(cont.M): # generate the coefficient matrix column-wise
        tempy[1:cont.N,j] = np.asarray(np.transpose(-(TD2[0,j] * (TD[cont.N,cont.N]*TD[1:cont.N,0]-TD[cont.N,0]*TD[1:cont.N,cont.N])/TD00_NN_TD0N_N0) \
                          + (TD2[cont.N,j] * (TD[0,cont.N]*TD[1:cont.N,0]-TD[0,0]*TD[1:cont.N,cont.N])/TD00_NN_TD0N_N0)))
    ##tempy[1:cont.M,1:cont.N] = (TD2[1:cont.M,1:cont.N] + tempy[1:cont.M,1:cont.N])
    tempy[1:cont.M,1:cont.N] = cont.asp**2*(TD2[1:cont.M,1:cont.N] + tempy[1:cont.M,1:cont.N])
    
    eigen_val_phi1[1:cont.M], eigen_vec_phi1[1:cont.M,1:cont.M] = np.linalg.eig(tempx[1:cont.M,1:cont.M])
    eigen_val_phi2[1:cont.N], eigen_vec_phi2[1:cont.N,1:cont.N] = np.linalg.eig(tempy[1:cont.N,1:cont.N])
    
    return eigen_vec_phi1, eigen_val_phi1, eigen_vec_phi2, eigen_val_phi2


'''Helmholtz Equation'''

def Helmholtz_solver(RHS,sigma,eig_val1,eig_vec1,eig_val2,eig_vec2):
    tol = 1e-10
    tempX = np.zeros([cont.M+1,cont.N+1])
    tempH = np.zeros([cont.M+1,cont.N+1])
    X = np.zeros([cont.M+1,cont.N+1])
    
    tempH[1:cont.M,1:cont.M] = np.linalg.solve(eig_vec1[1:cont.M,1:cont.M],RHS[1:cont.M,1:cont.M])
    tempH[1:cont.M,1:cont.M] = np.matmul(tempH[1:cont.M,1:cont.M],eig_vec2[1:cont.M,1:cont.M])
    
    for j in range(1,cont.N):
        for i in range(1,cont.M):
            if abs(eig_val1[i])<tol and abs(eig_val2[j])<tol and abs(sigma)<tol:
                X[i,j] = 0.0
            else:
                X[i,j] = tempH[i,j]/(eig_val1[i]+eig_val2[j]-sigma)
 
    tempX[1:cont.M,1:cont.N] = np.linalg.solve(np.transpose(eig_vec2[1:cont.M,1:cont.N]),np.transpose(X[1:cont.M,1:cont.N]))
    tempX[1:cont.M,1:cont.N] = np.transpose(tempX[1:cont.M,1:cont.N])
    X[1:cont.M,1:cont.N] = np.matmul(eig_vec1[1:cont.M,1:cont.N],tempX[1:cont.M,1:cont.N])
  
    return X


'''Solve for T, P, U, V, Phi'''

## Temperature
def T(U0,V0,T0,U1,V1,T1):
    Tnew = np.zeros([cont.M+1,cont.N+1])
    RHS = RHS_T(U0,V0,T0,U1,V1,T1)
    sigma = 3.0*cont.Pr/(2.0*cont.dt)
    eig_vec1, eig_val1, eig_vec2, eig_val2 = eigens_T()
    Tnew=Helmholtz_solver(RHS,sigma,eig_val1,eig_vec1,eig_val2,eig_vec2)
    for j in range(1,cont.M):      # c1 and c2 need to be defined
        Tnew[0,j] = np.dot(D[cont.M,0]*D[0,1:cont.M]-D[0,0]*D[cont.M,1:cont.M],Tnew[1:cont.M,j])/c1
        Tnew[-1,j]= np.dot(D[cont.M,cont.M]*D[0,1:cont.M]-D[0,cont.M]*D[cont.M,1:cont.M],Tnew[1:cont.M,j])/c2
  # update top and bottom boundaries    
    Tnew[:,0] = 0.5
    Tnew[:,-1] = -0.5
    
    Tnew = np.asmatrix(Tnew)
    return Tnew


## Pressure
def Pressure(U0,V0,T0,U1,V1,T1,T2):
    P = np.asmatrix(np.zeros([cont.M+1,cont.N+1]))
    rhs1, rhs2, RHS = RHS_P(U0,V0,T0,U1,V1,T1,T2)
    eig_vec1, eig_val1, eig_vec2, eig_val2 = eigens_P()
    sigma=0.0
    P=Helmholtz_solver(RHS,sigma,eig_val1,eig_vec1,eig_val2,eig_vec2)
    P[:,0] = 0
    P[:,cont.M] = 0
    for j in range(1,cont.M):
        P[0,j]=((D[cont.M,cont.M]*0.5*rhs1[0,j]- D[0,cont.M]*0.5*rhs1[cont.M,j]) 
                 -np.dot(D[cont.M,cont.M]*D[0,1:cont.N]-D[0,cont.M]*D[cont.M,1:cont.M], P[1:cont.M,j])) / c1
        
        P[cont.M,j]=((D[cont.M,0]*0.5*rhs1[0,j]- D[0,0]*0.5*rhs1[cont.M,j]) 
                 -np.dot(D[cont.N,0]*D[0,1:cont.N]-D[0,0]*D[cont.N,1:cont.N], P[1:cont.M,j])) / c2

    for i in range(0,cont.M+1):
        P[i,0]     =( (TD[cont.M,cont.M]*0.5*rhs2[i,0] -TD[cont.M,0]*0.5*rhs2[i,cont.M])
                 - np.dot(P[i,1:cont.M], TD[cont.M,cont.M]*TD[1:cont.M,0]-TD[cont.M,0]*TD[1:cont.M,cont.M]) )/c3
        
        P[i,cont.M]=( (TD[0,cont.M]     *0.5*rhs2[i,0] -TD[0,0]    *0.5*rhs2[i,cont.M])
                 - np.dot(P[i,1:cont.M], TD[0,cont.M]     *TD[1:cont.M,0]-TD[0,0]    *TD[1:cont.M,cont.M]) )/c4                     
    P = np.asmatrix(P)
    return P


## Velocity
def Velocity(U0,V0,T0,U1,V1,T1,T2,P):
    U = np.asmatrix(np.zeros([cont.M+1,cont.N+1]))
    V = np.asmatrix(np.zeros([cont.M+1,cont.N+1]))
    eig_vec1, eig_val1, eig_vec2, eig_val2 = eigens_vel()
    RHSU,RHSV = RHS_U(U0,V0,T0,U1,V1,T1,T2,P)       
    sigma = 3.0*cont.Pr/(2.0*cont.dt)
    
    U=Helmholtz_solver(RHSU,sigma,eig_val1,eig_vec1,eig_val2,eig_vec2)
    V=Helmholtz_solver(RHSV,sigma,eig_val1,eig_vec1,eig_val2,eig_vec2)
    
    U = np.asmatrix(U)
    V = np.asmatrix(V)
    
    # impose exact boundary velocity    
    U[:,0] = Uexact[:,0] 
    V[:,0] = Vexact[:,0]
    U[:,cont.M] = Uexact[:,cont.M]
    V[:,cont.M] = Vexact[:,cont.N]
    U[0,:] = Uexact[0,:]  
    V[0,:] = Vexact[0,:]
    U[cont.N,:] = Uexact[cont.N,:]
    V[cont.N,:] = Vexact[cont.N,:]
    
    return U, V

## Phi

def correction(U,V,P):
    RHS = np.asmatrix(np.zeros([cont.M+1,cont.N+1]))
    phi = np.asmatrix(np.zeros([cont.M+1,cont.N+1]))
    
    RHS = RHS_phi(U,V)
    eig_vec1, eig_val1, eig_vec2, eig_val2 = eigens_phi()
    sigma = 0.0
    
    phi=Helmholtz_solver(RHS,sigma,eig_val1,eig_vec1,eig_val2,eig_vec2)
    
    phi[:,0] = 0
    phi[:,cont.M] = 0
    
    for j in range(1,cont.M):  # update the boundary of phi
        phi[0,j] = -np.dot(D[cont.N,cont.N]*D[0,1:cont.N]-D[0,cont.N]*D[cont.N,1:cont.N],phi[1:cont.M,j]) /c1
        phi[cont.N,j] = -np.dot(D[cont.M,0]*D[0,1:cont.M]-D[0,0]*D[cont.M,1:cont.M],phi[1:cont.M,j]) /c2
        
    for j in range(0,cont.M+1):
        phi[j,0] = -np.dot(phi[j,1:cont.M], TD[cont.M,cont.M]*TD[1:cont.M,0]-TD[cont.M,0]*TD[1:cont.M,cont.M]) /c3
        phi[j,cont.M] = -np.dot(phi[j,1:cont.M], TD[0,cont.M]*TD[1:cont.M,0]-TD[0,0]*TD[1:cont.M,cont.M]) /c4
    
            
    P2 = P + 3*(cont.Pr/(2*cont.dt))*phi
    U2 = U - 2*D *phi
    V2 = V - 2*phi *TD
            
    return U2, V2, P2, phi  

'''Energy'''
def int_vector():
    W = np.zeros([cont.M+1])
    W[0] = 1.0/((cont.M**2) - 1.0)
    W[-1] = W[0]
    for j in range(1,cont.M):
        for i in range(1,int((cont.M/2))):
            W[j] = W[j] + 2.0/(1.0-4.0*(i**2)) * np.cos(2.0*i*j*np.pi/float(cont.M))
        W[j] = 1.0 + W[j] + np.cos(j*np.pi)/(1.0-(float(cont.M)**2.0))
        W[j] = (2.0/float(cont.M)) * W[j]
                              
    return W

def energy(U2,V2):
    W = int_vector()
    U2 = np.asarray(U2)
    V2 = np.asarray(V2)
    
    E = 0.25*(np.dot(W,np.matmul(W,(U2*U2 + V2*V2)))/cont.Re**2)
    
    return E