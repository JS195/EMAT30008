import numpy as np
import matplotlib.pyplot as plt
from Week_19 import matrix_build, boundary_conditions
from Week_20 import time_grid
import scipy.interpolate

def b_euler(N, a, b, gamma1, gamma2, D):
    dt, dx, t, N_time, x_int = time_grid(N, a, b, D)
    C = 0.1/dx**2
    A_matrix = matrix_build(N, D)
    b_matrix = A_matrix @ boundary_conditions(N, gamma1, gamma2)
    identity = np.identity(N-1)
    U = np.zeros((N_time + 1, N-1))
    U[0,:] = np.sin(np.pi*x_int)

    for i in range(0, N_time-1):
        U[i+1,:] = np.linalg.solve(identity + C*A_matrix, U[i,:] + C*b_matrix)
    
    return U, x_int

def crank(N, a, b, gamma1, gamma2, D):
    dt, dx, t, N_time, x_int = time_grid(N, a, b, D)
    C = 0.1/dx**2
    A_matrix = matrix_build(N, D)
    b_matrix = A_matrix @ boundary_conditions(N, gamma1, gamma2)
    identity = np.identity(N-1)
    U = np.zeros((N_time + 1, N-1))
    U[0,:] = np.sin(np.pi*x_int)

    for i in range(0, N_time-1):
        U[i+1,:] = np.linalg.solve(identity - C/2 * A_matrix, (identity + C/2 * A_matrix) @ U[i,:] + C * b_matrix)
    
    return U, x_int

def interp_comparison(x, t, N, a, b, gamma1, gamma2, D):
    print('true sol = ', np.exp(-0.2*np.pi**2))
    U, x_int = crank(N, a, b, gamma1, gamma2, D)
    y_interp = scipy.interpolate.interp1d(U[t,:], x_int, kind='linear')
    print('crank nichol = ', y_interp(x))
    U, x_int = b_euler(N, a, b, gamma1, gamma2, D)
    y_interp = scipy.interpolate.interp1d(U[t,:], x_int, kind='linear')
    print('backwards euler = ', y_interp(x))

def main():
    # Defining some initial variables
    N = 50
    a = 0
    b = 1
    gamma1 = 0.0
    gamma2 = 0.0
    D = 0.1

    # Part 1, backwards Euler demonstration
    # Plot the solution for timestep t = 0
    U, x_int = b_euler(N, a, b, gamma1, gamma2, D)
    print(U)
    plt.plot(x_int, U[0,:], 'ro', label='solution for t=0')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u')
    plt.xlim(a, b)
    plt.ylim(a, b)
    plt.show()

    # Part 2, crank nicholson method
    U, x_int = crank(N, a, b, gamma1, gamma2, D)
    plt.plot(x_int, U[0,:], 'bo', label='solution for t=0')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u')
    plt.xlim(a, b)
    plt.ylim(a, b)
    plt.show()

    # Part 3, comparing Euler and crank methods
    interp_comparison(0.5, 2, N, a, b, gamma1, gamma2, D)

if __name__ == "__main__":
    main()
