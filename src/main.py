## simulations
import numpy as np
import input_parameters as cont
import differentiation_matrix as diff
import func_modules

''' initialize Chebyshev differentiation matrix '''
x,y = diff.xy_grid()

''' Call initialized U,V,T matrices '''
## U0,U1,U2 at t = 0
U0 = diff.U_0(x)
U0[:,0] = 0
U1 = U0
U2 = U0

## V0,V1,V2 at t = 0
V0 = diff.V_0()
V1 = V0
V2 = V0

## T0,T1,T2 at t = 0
T0 = np.asmatrix(np.zeros([cont.M+1,cont.M+1]))
T1 = diff.T_0()
for j in range(1,cont.M): 
    T1[:,j] = y[j]
T2 = T0

#E = []
filename ='Re_'+ str(cont.Re)+'_Ri'+str(cont.Ri)+'_M64xM64'+'.dat'
fid = open(filename, "w")
for i in range(1,cont.num_steps+1):
    ''' Solve first instance '''

    ## Tempearture 
    T2 = func_modules.T(U0,V0,T0,U1,V1,T1)

    ## Pressure
    P = func_modules.Pressure(U0,V0,T0,U1,V1,T1,T2)


    ## Velocity
    U,V = func_modules.Velocity(U0,V0,T0,U1,V1,T1,T2,P)

    ## Phi, P2, U2, V2
    U2, V2, P2, phi = func_modules.correction(U,V,P)


    ## Energy
    E = func_modules.energy(U2,V2)
    fid.write(str([cont.dt*i,E])+'\n')

    ## Next Step

    ## Temperature
    T0 = T1
    T1 = T2


    ## Velocity
    U0 = U1
    U1 = U2

    V0 = V1
    V1 = V2

    #E.append(func_modules.energy(U2,V2))
    
    
fid.close()
    

