import scipy.interpolate
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Week_19 import *
from ODEsolver import RK4_step

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

def explicit_euler(N, D, gamma1, gamma2, a, b, dt, dx, t, N_time, x_int):
    A_matrix = matrix_build(N,D)
    b_matrix = A_matrix @ boundary_conditions(N,gamma1,gamma2)

    U = np.zeros((N_time + 1, N-1))
    U[0,:] = np.sin(np.pi*(x_int-a)/(b-a))
    
    for i in range(0,N_time-1):
        U[i+1,:] = U[i,:] + (dt*D/dx**2)*(A_matrix@U[i,:]+b_matrix) 

    u_true = np.zeros((N_time+1, N-1))
    for n in range(0, N_time):
        u_true[n,:] = np.exp(-(b-a)**2*D*np.pi**2*t[n]) * np.sin(np.pi*(x_int-a)/(b-a))
    
    return U, u_true

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

def animate_solution(U, u_true, x_int, N_time):
    fig, ax = plt.subplots()
    ax.set_ylim(0,1)
    ax.set_xlabel(f'$x$')
    ax.set_ylabel(f'$u(x,t)$')

    line, = ax.plot(x_int, u_true[0, :], label='True solution')
    line2, = ax.plot(x_int, U[0, :], 'ro', label='Numerical solution')

    def animate(i):
        line.set_data((x_int, u_true[i,:]))
        line2.set_data((x_int, U[i,:]))
        return line, line2

    ani = animation.FuncAnimation(fig, animate, frames=N_time, blit=False, interval=50)
    ax.legend()
    plt.show()

def implicit_euler(N, a, b, gamma1, gamma2, D):
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
    U, x_int = implicit_euler(N, a, b, gamma1, gamma2, D)
    y_interp = scipy.interpolate.interp1d(U[t,:], x_int, kind='linear')
    print('backwards euler = ', y_interp(x))

def main():
    N=20
    a=0
    b=1
    gamma1=0.0
    gamma2=0.0
    D = 1

    dt, dx, t, N_time, x_int = time_grid(N, a, b, D)
    U, u_true = explicit_euler(N, D, gamma1, gamma2, a, b, dt, dx, t, N_time, x_int)
    animate_solution(U, u_true, x_int, N_time)

    # Defining some initial variables
    N = 50
    a = 0
    b = 1
    gamma1 = 0.0
    gamma2 = 0.0
    D = 0.1

    # Part 1, backwards Euler demonstration
    # Plot the solution for timestep t = 0
    U, x_int = implicit_euler(N, a, b, gamma1, gamma2, D)
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