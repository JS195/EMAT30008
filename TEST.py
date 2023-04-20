import numpy as np
from Week_19 import finite_grid, matrix_build, boundary_conditions

def time_grid(N, a, b, D, C=0.49):
    grid = finite_grid(N, a, b)
    dx = grid[1]
    x_int = grid[2]
    dt = C*dx**2/D
    t_final = 1
    N_time = int(np.ceil(t_final/dt))
    t = dt*np.arange(N_time)
    return dt, dx, t, N_time, x_int

def explicit_euler(N, D, gamma1, gamma2, a, b, C, IC, true_sol='None'):
    dt, dx, t, N_time, x_int = time_grid(N, a, b , D, C)
    A_matrix = matrix_build(N,D)
    b_matrix = A_matrix @ boundary_conditions(N,gamma1,gamma2)
    U = np.zeros((N_time + 1, N-1))
    U[0,:] = IC(x_int, a, b)
    for i in range(0,N_time-1):
        U[i+1,:] = U[i,:] + (dt*D/dx**2)*(A_matrix@U[i,:]+b_matrix) 
    if true_sol != 'None':
        u_true = np.zeros((N_time+1, N-1))
        for n in range(0, N_time):
            u_true[n,:] = true_sol(t, n, x_int, a, b, D)
        return U, u_true
    else:
        return U

def initial_condition1(x_values, a, b):
    return np.sin(np.pi*(x_values-a)/(b-a))

def true_solution(t, n, x_int, a, b, D):
    return np.exp(-(b-a)**2*D*np.pi**2*t[n]) * np.sin(np.pi*(x_int-a)/(b-a))

Explicit_ans, true_ans = explicit_euler(N=20, D=1, gamma1=0.0, gamma2=0.0, a=0, b=1, C=0.49, IC=initial_condition1, true_sol=true_solution)
