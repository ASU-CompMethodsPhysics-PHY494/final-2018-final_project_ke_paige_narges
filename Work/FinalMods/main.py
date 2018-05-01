## simulations
import numpy as np
import input_parameters as cont
import differentiation_matrix as diff
import temporal_schemes

## boundary conditions

## Call initialized U,V,T matrices
U0 = diff.U_0()
U1 = U0
V0 = diff.V_0()
V1 = V0
T0 = diff.T_0()
T1 = T0
T2 = T1

## U orginial BC
U0j = U0[0,:]
Umj = U0[-1,:]
Ui0 = U0[:,0]
Uim = U0[:,-1]

## V original BC
V0j = V0[0,:]
Vmj = V0[-1,:]
Vi0 = V0[:,0]
Vim = V0[:,-1]

NL(U0,V0,T0)
A,B,C = temporal_schemes.LHS_T()
np.shape(A),np.shape(B),np.shape(C)


temporal_schemes.NL(U0,V0,T0)

## Step 1 solve for T*
temporal_schemes.RHS_T(U0,V0,T0,U1,V1,T1)
A,B,C = temporal_schemes.LHS_T()
np.shape(A),np.shape(B),np.shape(C)

## Step 2 solve for P*
RHSP = temporal_schemes.RHS_P(U0,V0,T0,U1,V1,T1,T2)
np.shape(RHSP)
A,B,C = temporal_schemes.LHS_P()
np.shape(A), np.shape(B), np.shape(C)


## Step 3 solve for U*,V*
RHSU, RHSV = temporal_schemes.RHS_U(U0,V0,T0,U1,V1,T1,T2,P)
np.shape(RHSU), np.shape(RHSV)
A,B,C = temporal_schemes.LHS_U()
np.shape(A), np.shape(B), np.shape(C)

## Step 4 solve for Z*
RHSZ = temporal_schemes.RHS_Z(U1,V1)
np.shape(RHSZ)
A,B,C = temporal_schemes.LHS_Z()
np.shape(A), np.shape(B), np.shape(C)

## Step 5: Use P*, U*, V*, Z*
P2,U2,V2 = temporal_schemes.PUV(P,U,V,Z)
