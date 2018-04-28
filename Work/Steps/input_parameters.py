M = 64                    # number of grids on x direction 
N = 64                    # number of grids on y direction
Re= 1000                  # Reynolds number
Ri= 0.01                  # Richardson number
asp= 2                     # aspect ratio
dt = 1.0d-3                # time step
num_steps=10000                 # total time iterations
its   = 1                    # its write time series every its time steps
igraph=1000                  # igraph write restart every igraph time steps
ibegin= 0                     # ibegin =0 start with 0 initial condition 
                      #        =1 continue restart and keep t
                      #        =2 continue restart with different parameters 
