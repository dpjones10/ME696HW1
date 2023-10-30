# ME 696 Computational Transport Phenomena 
# Drake Jones

# HW 1_4 2D Heat Conduction

import numpy as np
import matplotlib.pyplot as plt
import time
from Cond_1D import Cond_1D

start = time.time()     # record start time

# physical constants
L = 1.0             # length of domain [m]
H = 0.25            # height of domain [m]
k = 1.0             # thermal conductivity [W/mK]
h = 100.0           # convection heat transfer coefficeint [W / m^2K]
T1s = 100.0         # temp at left boundary [K]
T_inf = 25.0        # ambient air temperature [K]
k_i = 1.0 / k       # reciprocl of k to prevent repeated division

# numerical parameters
nx = 64             # number of cells in x direction
ny = int(nx/2)           # number of cells in y direction
dx = L / nx         # grid size in x direction
dy = H / ny         # grid size in y direction
dx_i = 1.0 / dx     # recipricol of dx to prevent repeated division
dy_i = 1.0 / dy     # recipricol of dy to prevent repeated division

# initializing arrays
x = np.zeros([nx,ny])
y = np.zeros([nx,ny])
q_dot = np.zeros([nx,ny])
A = np.zeros([nx*ny,nx*ny])
b = np.zeros(nx*ny)

for j in range(ny):
    for i in range(nx):

        ij = i + j*nx           # linearized index for cell (i,j)
        im1j = i-1 + j*nx       # linearized index for cell (i-1,j)
        ip1j = i+1 + j*nx       # linearized index for cell (i+1,j)
        ijm1 = i + (j-1)*nx     # linearized index for cell (i,j-1)
        ijp1 = i + (j+1)*nx     # linearized index for cell (i,j+1) 
        
        x[i][j] = dx*i + (dx/2)     # cell centered x coordinate
        y[i][j] = dy*j + (dy/2)     # cell centered y coordinate

        q_dot[i][j] = 10.0*x[i][j] + 5.0        # heat generation density

        if ((i > 0) and (i < nx-1) and (j > 0) and (j < ny-1)):
            
            A[ijm1][ij] = -dx*dy_i        # populating matrix for internal cells
            A[im1j][ij] = -dy*dx_i        
            A[ij][ij] = 2.0*(dy*dx_i + dx*dy_i)
            A[ip1j][ij] = -dy*dx_i
            A[ijp1][ij] = -dx*dy_i

            b[ij] = dx*dy*k_i*q_dot[i][j]       # populating RHS for internal cells

        elif ((i == 0) and (j > 0) and (j < ny-1)):
            
            A[ijm1][ij] = -dx*dy_i      # populating matrix for x = 0 BC (Dirichlet: T[i-1/2][j] = T1s)
            A[ij][ij] = 3.0*dy*dx_i + 2.0*dx*dy_i
            A[ip1j][ij] = -dy*dx_i
            A[ijp1][ij] = -dx*dy_i

            b[ij] = dx*dy*k_i*q_dot[i][j] + 2.0*dy*dx_i*T1s     # populating RHS for x = 0 BC (Dirichlet: T[i-1/2][j] = T1s)
        
        elif ((i == nx-1) and (j > 0) and (j < ny-1)):
            
            A[ijm1][ij] = -dx*dy_i      # populating matrix for x = L bc (Mixed: q"_conduction = q"_convection)
            A[im1j][ij] = -dy*dx_i
            A[ij][ij] = dy*dx_i + dy*dx_i*(2.0*h / (h + 2.0*k*dx_i)) + 2.0*dx*dy_i
            A[ijp1][ij] = -dx*dy_i

            b[ij] = dx*dy*k_i*q_dot[i][j] + dy*dx_i*(2.0*h / (h + 2.0*k*dx_i))*T_inf      # populating RHS for x = L BC
        
        elif ((i > 0) and (i < nx-1) and (j == 0)):

            A[im1j][ij] = -dy*dx_i      # populating matrix for y = 0 BC (Neumann: dT[i][j-1/2]/dy = 0)
            A[ij][ij] = 2.0*dy*dx_i + dx*dy_i
            A[ip1j][ij] = -dy*dx_i
            A[ijp1][ij] = -dx*dy_i

            b[ij] = dx*dy*k_i*q_dot[i][j]       # populating RHS for y = 0 BC (Neumann: dT[i][j-1/2]/dy = 0)
        
        elif ((i > 0) and (i < nx-1) and (j == ny-1)):

            A[ijm1][ij] = -dx*dy_i      # populating matrix for y = H BC (Neumann: dT[i][j+1/2]/dy = 0)
            A[im1j][ij] = -dy*dx_i
            A[ij][ij] = 2.0*dy*dx_i + dx*dy_i
            A[ip1j][ij] = -dy*dx_i
            
            b[ij] = dx*dy*k_i*q_dot[i][j]       # populating RHS for y = H BC (Neumann: dT[i][j+1/2]/dy = 0)

        elif ((i == 0) and (j == 0)):

            A[ij][ij] = 3.0*dy*dx_i + dx*dy_i   # populating matrix for x = 0, y = 0 BC (T[i-1/2][j] = T1s; dT[i][j-1/2]/dy = 0)
            A[ip1j][ij] = -dy*dx_i
            A[ijp1][ij] = -dx*dy_i

            b[ij] = dx*dy*k_i*q_dot[i][j] + 2.0*dy*dx_i*T1s  # populating RHS for x = 0, y = 0 BC

        elif ((i == nx-1) and (j == 0)):
            
            A[im1j][ij] = -dy*dx_i      # populating matrix for x = L, y = 0 BC (q"_conduction = q"_convection; dT[i][j-1/2]/dy = 0)
            A[ij][ij] = dy*dx_i + dy*dx_i*(2.0*h / (h + 2.0*k*dx_i)) + dx*dy_i
            A[ijp1][ij] = -dx*dy_i

            b[ij] = dx*dy*k_i*q_dot[i][j] + dy*dx_i*(2.0*h / (h + 2.0*k*dx_i))*T_inf    # populating matrix for x = l, y = 0 BC

        elif ((i == 0) and (j == ny-1)):

            A[ijm1][ij] = -dx*dy_i      # populating matrix for x = 0, y = H BC (T[i-1/2][j] = T1s; dT[i][j+1/2]/dy = 0)
            A[ij][ij] = 3.0*dy*dx_i + dx*dy_i
            A[ip1j][ij] = -dy*dx_i
            
            b[ij] = dx*dy*k_i*q_dot[i][j] + 2.0*dy*dx_i*T1s     # populating RHS for x = 0, y = H BC

        elif ((i == nx-1) and (j == ny-1)):

            A[ijm1][ij] = -dx*dy_i      # populating matrix for x = L, y = H BC (q"_conduction = q"_convection; dT[i][j+1/2]/dy = 0)
            A[im1j][ij] = -dy*dx_i  
            A[ij][ij] = dy*dx_i + dy*dx_i*(2.0*h / (h + 2.0*k*dx_i)) + dx*dy_i

            b[ij] = dx*dy*k_i*q_dot[i][j] + dy*dx_i*(2.0*h / (h + 2.0*k*dx_i))*T_inf      # populating RHS for x = L, y = H BC


x = np.transpose(x)
y = np.transpose(y)
#A = np.transpose(A)     # transposing A because of matrix A was populated columnwise for python memory

T_2D = np.linalg.solve(A,b)
T_2D = T_2D.reshape(ny,nx)      # reshaping T matrix to match domain
 
T_H_2 = (T_2D[int(ny/2)-1] + T_2D[int(ny/2)]) / 2.0       # interpolating T to cell face at y = H/2
T_L_2 = (T_2D[:,int(nx/2)-1] + T_2D[:,int(nx/2)]) / 2.0   # interpolating T to cell face at x = L/2

T_1D = Cond_1D(L,k,h,T1s,T_inf,nx)      # 1D conduction solution
Err = np.abs(T_H_2 - T_1D)      # difference betwen 1D and 2D solutions

end = time.time()   # record end time
run_time = end - start

print("run_time =",run_time)

plt.figure()
plt.plot(x[0],T_1D)
plt.plot(x[0],T_H_2)
plt.xlabel("x [m]")
plt.ylabel("T [K]")
plt.title("1D vs 2D Conduction Comparison")
plt.legend(["$T_{1D}$","$T_{2D}$"])
plt.show()

plt.figure()
plt.plot(x[0],Err)
plt.xlabel("x [m]")
plt.ylabel("T [K]")
plt.title("1D vs 2D Temperature Difference")
plt.show()

plt.figure()
plt.plot(y[:,0],T_L_2)
plt.xlabel("y [m]")
plt.ylabel("T [K]")
plt.title("Temperature Profile at (L/2,y)")
plt.show()

plt.figure()
plt.contourf(x,y,T_2D,cmap="coolwarm")
plt.colorbar()
plt.ylim(0.0,0.25)
plt.axis('equal')
#plt.show()

