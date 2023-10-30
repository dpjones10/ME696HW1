# 1D conduction function

import numpy as np

def Cond_1D(L,k,h,T1s,T_inf,n):

    k_i = 1.0 / k       # reciprocl of k to prevent repeated division

    # numerical parameters
    dx = L / n              # cell size [m]
    dx_i = 1.0 / dx         # recipercol of cell size to limit repeated divison

    # initializing arrays
    x_n = np.zeros(n)         
    q_dot = np.zeros(n)
    A = np.zeros([n,n])
    b = np.zeros(n)
       
    for i in range(n):

        x_n[i] = dx*i + (dx/2.0)        # cell centered x values
        q_dot[i] = 10.0*x_n[i] + 5.0    # heat generation density

        if ((i > 0) and (i < n-1)):
            
            A[i][i-1] = -dx_i        # populating matrix for internal cells
            A[i][i] = 2.0*dx_i
            A[i][i+1] = -dx_i

            b[i] = q_dot[i]*k_i*dx     # populating RHS for internal cells

        elif (i == 0):
            
            A[i][i] = 3.0*dx_i     # populating matrix for x = 0 BC
            A[i][i+1] = -dx_i

            b[i] = q_dot[i]*k_i*dx + (2.0*dx_i)*T1s    # populating RHS for x = 0 BC

        else:

            A[i][i-1] = -dx_i       # populating matrix for x = L BC
            A[i][i] = ((2.0*h*dx_i / (h + 2.0*k*dx_i)) + dx_i)  # populating RHS for x = H BC

            b[i] = q_dot[i]*k_i*dx + (2.0*h*dx_i / (h + 2.0*k*dx_i))*T_inf
        
    T_n = np.linalg.solve(A,b)

    return T_n
