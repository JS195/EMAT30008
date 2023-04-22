import numpy as np
from PDEsolver import time_grid, animate_solution
from BVPsolver import matrix_build, boundary_conditions, finite_grid
from ExampleFunctions import linear_diffusion_IC
import matplotlib.pyplot as plt
import scipy.interpolate
from ExampleFunctions import linear_diffusion_IC

def time_grid(N, a, b, D, C=0.49):
    # C must be smaller than 0.5, else the system bugs out
    grid = finite_grid(N, a, b)
    dx = grid[1]
    x_int = grid[2]
    dt = C*dx**2/D
    t_final = 1
    N_time = int(np.ceil(t_final/dt))
    t = dt*np.arange(N_time)
    return dt, dx, t, N_time, x_int

def crank(N, gamma1, gamma2, D, N_time, x_int, dt, dx):
    C = D*dt/dx**2
    A_matrix = matrix_build(N, D)
    b_matrix = A_matrix @ boundary_conditions(N, gamma1, gamma2)
    identity = np.identity(N-1)
    U = np.zeros((N_time + 1, N-1))
    U[0,:] = np.sin(np.pi*x_int)

    for i in range(0, N_time-1):
        U[i+1,:] = np.linalg.solve(identity - C/2 * A_matrix, (identity + C/2 * A_matrix) @ U[i,:] + C * b_matrix)
    
    return U

N=100
a=0
b=1
gamma1=0.0
gamma2=0.0
D = 0.1

dt, dx, t, N_time, x_int = time_grid(N, a, b, D)
U_implicit = crank(N, gamma1, gamma2, D, N_time, x_int, 0.1, dx)
#animate_solution(U_explicit, u_true, x_int, N_time)
print('true sol = ', np.exp(-0.2*np.pi**2))
y_interp1 = scipy.interpolate.interp1d(U_implicit[2,:], x_int, kind='linear')
print('implicit euler = ', y_interp1(0.5))