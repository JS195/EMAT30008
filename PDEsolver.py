import scipy.interpolate
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Week_19 import finite_grid, matrix_build, boundary_conditions
from ODEsolver import RK4_step

def time_grid(N, a, b, D, C=0.49):
    """
    Constructs a time grid for a one-dimensional diffusion problem with constant diffusivity.
    
    :param N: The number of grid points in the spatial domain.
    :param a: The left endpoint of the spatial domain.
    :param b: The right endpoint of the spatial domain.
    :param D: The diffusivity of the medium.
    :param C: Optional. The dimensionless Courant-Friedrichs-Lewy (CFL) number, which controls the time step size. Default value is 0.49.
    
    :returns: A tuple containing five elements:
              - dt: The time step size.
              - dx: The spatial step size.
              - t: An array of time values.
              - N_time: The number of time steps.
              - x_int: An array of spatial grid points.
    """
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
    """
    Solves a one-dimensional diffusion problem using the explicit Euler method.

    :param N: The number of grid points in the spatial domain.
    :param D: The diffusivity of the medium.
    :param gamma1: The left boundary condition.
    :param gamma2: The right boundary condition.
    :param a: The left endpoint of the spatial domain.
    :param b: The right endpoint of the spatial domain.
    :param dt: The time step size.
    :param dx: The spatial step size.
    :param t: An array of time values.
    :param N_time: The number of time steps.
    :param x_int: An array of spatial grid points.
    
    :returns: A tuple containing two arrays:
              - U: The numerical solution of the diffusion problem, as a function of time and space.
              - u_true: The true solution of the diffusion problem, as a function of time and space.
    """
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
    """
    Solves a one-dimensional diffusion problem using the fourth-order Runge-Kutta method.

    :param N: The number of grid points in the spatial domain.
    :param D: The diffusivity of the medium.
    :param gamma1: The left boundary condition.
    :param gamma2: The right boundary condition.
    :param a: The left endpoint of the spatial domain.
    :param b: The right endpoint of the spatial domain.
    :param dt: The time step size.
    :param dx: The spatial step size.
    :param t: An array of time values.
    :param x_int: An array of spatial grid points.
    
    :returns: A tuple containing two arrays:
              - U: The numerical solution of the diffusion problem, as a function of time and space.
              - u_true: The true solution of the diffusion problem, as a function of time and space.
    """

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

def implicit_euler(N, a, b, gamma1, gamma2, D):
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
    U, x_int = crank(N, a, b, gamma1, gamma2, D)
    y_interp = scipy.interpolate.interp1d(U[t,:], x_int, kind='linear')
    print('crank nichol = ', y_interp(x))
    U, x_int = implicit_euler(N, a, b, gamma1, gamma2, D)
    y_interp = scipy.interpolate.interp1d(U[t,:], x_int, kind='linear')
    print('implicit euler = ', y_interp(x))

def main():
    N=20
    a=0
    b=1
    gamma1=0.0
    gamma2=0.0
    D = 1

    dt, dx, t, N_time, x_int = time_grid(N, a, b, D)
    U, u_true = explicit_euler(N, D, gamma1, gamma2, a, b, dt, dx, t, N_time, x_int)
    print(U)
    animate_solution(U, u_true, x_int, N_time)

    # Defining some initial variables
    N = 50
    a = 0
    b = 1
    gamma1 = 0.0
    gamma2 = 0.0
    D = 0.1

    # Part 1, implicit Euler demonstration
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