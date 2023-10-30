# ME 696 Computational Transport Phenomena 
# Drake Jones

# HW 1_2 1D Heat Conduction

import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()     # record start time

# physical constants
L = 1.0             # length of domain [m]
k = 1.0             # thermal conductivity [W/mK]
h = 100.0           # convection heat transfer coefficeint [W / m^2K]
T1s = 100.0         # temp at left boundary [K]
T_inf = 25.0        # ambient air temperature [K]

k_i = 1.0 / k       # reciprocl of k to prevent repeated division
C1 = ((h/k)*((5.0/3.0)*L**3 + (5.0 / 2.0)*L**2) + 5.0*L**2 + 5.0*L - h*(T1s - T_inf)) / (h*L + k)   # integration constant

n = [16, 32, 64, 128]                 # number of cells

L_inf = np.zeros(len(n))
L1 = np.zeros(len(n))
L2 = np.zeros(len(n))

for q in range(len(n)):
    
    dx = L / n[q]           # cell size [m]
    dx_i = 1.0 / dx         # recipercol of cell size to limit repeated divison

    # initializing arrays
    x_n = np.zeros(n[q])         
    q_dot = np.zeros(n[q])
    T_a = np.zeros(n[q])
    A = np.zeros([n[q],n[q]])
    b = np.zeros(n[q])
        
    for i in range(n[q]):

        x_n[i] = dx*i + (dx/2.0)        # cell centered x coordinate
        q_dot[i] = 10.0*x_n[i] + 5.0    # heat generation density

        T_a[i] = -k_i*((5.0/3.0)*x_n[i]**3.0 + (5.0 / 2.0)*x_n[i]**2.0) + C1*x_n[i] + T1s

        if ((i > 0) and (i < n[q]-1)):
            
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
            A[i][i] = dx_i*(1.0 + (2.0*h / (h + 2.0*k*dx_i)))

            b[i] = q_dot[i]*k_i*dx + (2.0*h*dx_i / (h + 2.0*k*dx_i))*T_inf    # populating RHS for x = L BC
        
    
    T_n = np.linalg.solve(A,b)
    

    L_inf[q] = np.max(np.abs(T_n-T_a)) / T_a[np.argmax(np.abs(T_n-T_a))]      # L infinity error

    for i in range(n[q]):
        L1[q] = L1[q] + np.abs(T_a[i] - T_n[i]) / T_a[i]
        L2[q] = L2[q] + ((T_a[i] - T_n[i]) / T_a[i])**2

    L1[q] = L1[q] / n[q]
    L2[q] = np.sqrt(L2[q]) / np.sqrt(n[q])

end = time.time()   # record end time
run_time = end - start

print("run_time =",run_time)

plt.figure()
plt.plot(x_n,T_a,'r')
plt.plot(x_n,T_n,'b')
plt.xlabel("x [m]")
plt.ylabel("T [K]")
plt.title("Temperature Profile with Non-Uniform Heat Generation")
plt.legend(["$T_{Analytical}$","$T_{Numerical}$"])

plt.figure()
plt.loglog(n,L_inf,"-o")
plt.loglog(n,L1,"-o")
plt.loglog(n,L2,"-o")
plt.xlabel("N")
plt.ylabel("Error")
plt.title("Error Calculation Comarison")
plt.legend(["$L^{\infty}$","$L^1$","$L^2$"])

plt.show()