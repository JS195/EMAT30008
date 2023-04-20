import scipy.interpolate
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Week_19 import finite_grid, matrix_build, boundary_conditions
from ODEsolver import RK4_step
from Functions_and_ODEs import linear_diffusion_IC, linear_diffusion_true_sol

def time_grid(a, b, N, D, C=0.49):
    grid = finite_grid(N, a, b)
    dx = grid[1]
    x_int = grid[2]
    dt = C*dx**2/D
    t_final = 1
    N_time = int(np.ceil(t_final/dt))
    t = dt*np.arange(N_time)
    return dt, dx, t, N_time, x_int

def true_solution(a, b, N, D, C, true_sol):
    dt, dx, t, N_time, x_int = time_grid(a, b, N, D, C)
    u_true = np.zeros((N_time+1, N-1))
    for n in range(0, N_time):
        u_true[n,:] = true_sol(t, n, x_int, a, b, D)
    return u_true, N_time, x_int

def explicit_euler(a, b, gamma1, gamma2, N, D, C, IC):
    dt, dx, t, N_time, x_int = time_grid(a, b, N, D, C)
    A_matrix = matrix_build(N,D)
    b_matrix = A_matrix @ boundary_conditions(N,gamma1,gamma2)
    U = np.zeros((N_time + 1, N-1))
    U[0,:] = IC(x_int, a, b)
    for i in range(0,N_time-1):
        U[i+1,:] = U[i,:] + (dt*D/dx**2)*(A_matrix@U[i,:]+b_matrix) 
    return U, N_time, x_int

def RK4_method(a, b, gamma1, gamma2, N, D, C, IC):
    dt, dx, t, N_time, x_int = time_grid(a, b, N, D, C)
    A_matrix = matrix_build(N,D)
    b_matrix = A_matrix @ boundary_conditions(N,gamma1,gamma2)
    U = np.zeros((len(t), N-1))
    U[0,:] = IC(x_int, a, b)
    def f(U, t):
        return D/dx**2 * (A_matrix @ U + b_matrix)
    for i in range(len(t)-1):
        U[i+1,:], _ = RK4_step(f, U[i,:], t[i], dt)
    return U, N_time, x_int

def animate_solution(U, u_true, x_int, N_time):
    """
    Animates the numerical and true solutions to a PDE.

    :param U: The numerical solution to the PDE, obtained from a numerical method.
    :param u_true: The true solution to the PDE, obtained from an analytical method.
    :param x_int: The interval on which the solutions are defined.
    :param N_time: The number of time steps in the solution.
    
    :returns: None, but produces an animation of the numerical and true solutions.
    """
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

def implicit_euler(a, b, gamma1, gamma2, N, D, C, IC):
    """
    Animates the numerical and true solutions to a PDE using implicit Euler method.

    :param N: The number of grid points.
    :param a: The left endpoint of the interval.
    :param b: The right endpoint of the interval.
    :param gamma1: The boundary condition at the left endpoint.
    :param gamma2: The boundary condition at the right endpoint.
    :param D: The diffusion coefficient.
    
    :returns: A tuple (U, x_int) containing the numerical solution to the PDE obtained from 
          implicit Euler method and the interval on which the solution is defined.
    """
    dt, dx, t, N_time, x_int = time_grid(a, b, N, D, C)
    C = 0.1/dx**2
    A_matrix = matrix_build(N, D)
    b_matrix = A_matrix @ boundary_conditions(N, gamma1, gamma2)
    identity = np.identity(N-1)
    U = np.zeros((N_time + 1, N-1))
    U[0,:] = IC(x_int, a, b)

    for i in range(0, N_time-1):
        U[i+1,:] = np.linalg.solve(identity + C*A_matrix, U[i,:] + C*b_matrix)
    
    return U, N_time, x_int

def crank(a, b, gamma1, gamma2, N, D, C, IC):
    """
    Animates the numerical and true solutions to a PDE using the Crank-Nicolson method.
    :param N: The number of grid points.
    :param a: The left endpoint of the interval.
    :param b: The right endpoint of the interval.
    :param gamma1: The boundary condition at the left endpoint.
    :param gamma2: The boundary condition at the right endpoint.
    :param D: The diffusion coefficient.
    
    :returns: A tuple (U, x_int) containing the numerical solution to the PDE obtained from 
          the Crank-Nicolson method and the interval on which the solution is defined.
    """
    dt, dx, t, N_time, x_int = time_grid(a, b, N, D, C)
    C = 0.1/dx**2
    A_matrix = matrix_build(N, D)
    b_matrix = A_matrix @ boundary_conditions(N, gamma1, gamma2)
    identity = np.identity(N-1)
    U = np.zeros((N_time + 1, N-1))
    U[0,:] = IC(x_int, a, b)
    for i in range(0, N_time-1):
        U[i+1,:] = np.linalg.solve(identity - C/2 * A_matrix, (identity + C/2 * A_matrix) @ U[i,:] + C * b_matrix
    return U, N_time, x_int


def interp_comparison(a, b, gamma1, gamma2, N, D, x, t):
    """
    Compares the interpolated numerical solutions at a specific time to the true solution to a PDE
    obtained from the Crank-Nicolson and implicit Euler methods.

    :param x: The point at which to interpolate the numerical solution.
    :param t: The time step at which to interpolate the numerical solution.
    :param N: The number of grid points.
    :param a: The left endpoint of the interval.
    :param b: The right endpoint of the interval.
    :param gamma1: The boundary condition at the left endpoint.
    :param gamma2: The boundary condition at the right endpoint.
    :param D: The diffusion coefficient.
    
    :returns: None, but prints the true solution and the interpolated numerical solutions 
          obtained from the Crank-Nicolson and implicit Euler methods at the specified time step.
    """
    print('true sol = ', np.exp(-0.2*np.pi**2))
    U1, N_time1, x_int1 = crank(a=0, b=1, gamma1=0.0, gamma2=0.0, N=50, D=0.1, C=0.49, IC=linear_diffusion_IC)
    y_interp1 = scipy.interpolate.interp1d(U1[t,:], x_int1, kind='linear')
    print('crank nichol = ', y_interp1(x))
    U2, N_time2, x_int2 = implicit_euler(a=0, b=1, gamma1=0.0, gamma2=0.0, N=50, D=0.1, C=0.49, IC=linear_diffusion_IC)
    y_interp2 = scipy.interpolate.interp1d(U2[t,:], x_int2, kind='linear')
    print('implicit euler = ', y_interp2(x))

def main():
    U, N_time, x_int = RK4_method(a=0, b=1, gamma1 = 0.0, gamma2=0.0, N=20, D=1, C=0.49, IC=linear_diffusion_IC)
    u_true = true_solution(a=0, b=1, N=20, D=1, C=0.49, true_sol=linear_diffusion_true_sol)[0]
    animate_solution(U, u_true, x_int, N_time)

    # Part 1, implicit Euler demonstration
    # Plot the solution for timestep t = 0
    U, N_time, x_int = implicit_euler(a=0, b=1, gamma1=0.0, gamma2=0.0, N=50, D=0.1, C=0.49, IC=linear_diffusion_IC)
    print(U)
    plt.plot(x_int, U[0,:], 'ro', label='solution for t=0')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u')
    plt.show()

    # Part 2, crank nicholson method
    U, N_time, x_int = crank(a=0, b=1, gamma1=0.0, gamma2=0.0, N=50, D=0.1, C=0.49, IC=linear_diffusion_IC)
    plt.plot(x_int, U[0,:], 'bo', label='solution for t=0')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u')
    plt.show()

    # Part 3, comparing Euler and crank methods
    interp_comparison(a=0, b=1, gamma1=0.0, gamma2=0.0, N=20, D=1, x=0.5, t=0)
if __name__ == "__main__":
    main()