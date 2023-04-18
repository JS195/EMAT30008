from Week_19_matrices import matrix_build, boundary_conditions
from Week_20 import time_grid
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

def b_euler(N, a, b, gamma1, gamma2, D):
    dt, dx, t, N_time, x_int = time_grid(N, a, b)
    C = 0.1/dx**2
    A_matrix = matrix_build(N, D)
    b_matrix = A_matrix @ boundary_conditions(N, gamma1, gamma2)
    identity = np.identity(N-1)
    U = np.zeros((N_time + 1, N-1))
    U[0,:] = np.sin(np.pi*x_int)

    for i in range(0, N_time-1):
        U[i+1,:] = np.linalg.solve(identity + C*A_matrix, U[i,:] + C*b_matrix)
    
    return U, x_int

N = 20
a = 0
b = 1
gamma1 = 0.0
gamma2 = 0.0
D = 0.1

# Plot the solution at each time step
U, x_int = b_euler(N, a, b, gamma1, gamma2, D)
plt.plot(x_int, U[0,:], 'ro', label='solution for t=0')
plt.legend()
plt.xlabel('x')
plt.ylabel('u')
plt.xlim(a, b)
plt.ylim(a, b)
plt.show()
