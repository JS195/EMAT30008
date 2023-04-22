import numpy as np
from PDEsolver import time_grid, animate_solution
from BVPsolver import matrix_build, boundary_conditions, finite_grid
from ExampleFunctions import linear_diffusion_IC
import matplotlib.pyplot as plt
import scipy.interpolate
from ExampleFunctions import linear_diffusion_IC1

def explicit_euler(N, D, gamma1, gamma2, a, b, dt, dx, t, N_time, x_int):
    A_matrix = matrix_build(N,D)
    b_matrix = A_matrix @ boundary_conditions(N,gamma1,gamma2)

    U = np.zeros((N_time + 1, N-1))
    U[0,:] = linear_diffusion_IC1(x_int, a, b)
    
    for i in range(0,N_time-1):
        U[i+1,:] = U[i,:] + (dt*D/dx**2)*(A_matrix@U[i,:]+b_matrix) 

    return U

def true_sol_func(N, D, a, b, t, N_time, x_int):
    u_true = np.zeros((N_time+1, N-1))
    for n in range(0, N_time):
        u_true[n,:] = np.exp(-(b-a)**2*D*np.pi**2*t[n]) * np.sin(np.pi*(x_int-a)/(b-a))
    return u_true


def heat_equation_RK4(N, D, gamma1, gamma2, a, b, dt, dx, t, x_int):
    A_matrix = matrix_build(N,D)
    b_matrix = A_matrix @ boundary_conditions(N,gamma1,gamma2)

    U = np.zeros((len(t), N-1))
    U[0,:] = np.sin(np.pi*(x_int-a)/(b-a))
    
    def f(U, t):
        return D/dx**2 * (A_matrix @ U + b_matrix)
    
    for i in range(len(t)-1):
        U[i+1,:], _ = RK4_step(f, U[i,:], t[i], dt)

    u_true = np.zeros((len(t), N-1))
    for n in range(len(t)):
        u_true[n,:] = np.exp(-(b-a)**2*D*np.pi**2*t[n]) * np.sin(np.pi*(x_int-a)/(b-a))
    
    return U, u_true