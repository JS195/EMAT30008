import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.interpolate
from BVPsolver import matrix_build, dirichlet, finite_grid
from ODEsolver import RK4_step, plotter
from ExampleFunctions import linear_diffusion_IC1, linear_diffusion_IC2

def explicit_euler(N, D, gamma1, gamma2, a, b, dt, dx, t, N_time, x_int, IC):
    """
    Computes and plots the natural continuation of a system of ODEs as a function of a parameter.

    :param N: The number of spatial grid points.
    :param D: The diffusion coefficient.
    :param gamma1: The boundary condition value at x = 0.
    :param gamma2: The boundary condition value at x = 1.
    :param a: The left-end initial condition parameter.
    :param b: The right-end initial condition parameter.
    :param dt: The time step size.
    :param dx: The spatial step size.
    :param t: The initial time.
    :param N_time: The number of time steps.
    :param x_int: The spatial domain interval.

    :returns: A numpy array of shape (N_time+1, N-1) containing the solution at each time step for the explicit Euler method.
    """
    # Build the required matrices
    A_matrix = matrix_build(N,D)
    b_matrix = A_matrix @ dirichlet(N,gamma1,gamma2)

    # Initialize the U array with zeros
    U = np.zeros((N_time + 1, N-1))
    #Set the IC for U
    U[0,:] = IC(x_int, a, b)
    
    # Loop over each time step and compute solution
    for i in range(0,N_time-1):
        U[i+1,:] = U[i,:] + (dt*D/dx**2)*(A_matrix@U[i,:]+b_matrix) 

    return U

def RK4_method(N, D, gamma1, gamma2, a, b, dt, dx, t, x_int, IC):
    """
    Solves the PDE numerically using the fourth-order Runge-Kutta method.

    :param N: The number of spatial grid points.
    :param D: The diffusion coefficient.
    :param gamma1: The boundary condition value at x = 0.
    :param gamma2: The boundary condition value at x = 1.
    :param a: The left-end initial condition parameter.
    :param b: The right-end initial condition parameter.
    :param dt: The time step size.
    :param dx: The spatial step size.
    :param t: A numpy array containing the time values at which to compute the solution.
    :param x_int: A tuple containing the spatial domain interval (x_min, x_max).

    :returns: A numpy array of shape (len(t), N-1) containing the numerical solution of the PDE at the time values given in the input t.
    """
    # Construct the required matrices
    A_matrix = matrix_build(N,D)
    b_matrix = A_matrix @ dirichlet(N,gamma1,gamma2)

    # Initialize the solution array U with the initial condition
    U = np.zeros((len(t), N-1))
    U[0,:] = IC(x_int, a, b)
    
    def f(U, t):
        return D/dx**2 * (A_matrix @ U + b_matrix)
    
    # Solve using rk4
    for i in range(len(t)-1):
        U[i+1,:], _ = RK4_step(f, U[i,:], t[i], dt)
    
    return U

def animate_solution(U, u_true, x_int):
    """
    Animates the numerical solution of the PDE along with the true solution.

    :param U: A numpy array of shape (N_time+1, N-1) containing the numerical solution of the PDE.
    :param u_true: A numpy array of shape (N_time+1, N-1) containing the true solution of the PDE.
    :param x_int: A tuple containing the spatial domain interval (x_min, x_max).
    :param N_time: The number of time steps.

    :returns: None, but animates a graph.
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

    ani = animation.FuncAnimation(fig, animate, frames=1000, blit=False, interval=50)
    ax.legend()
    plt.show()

def time_grid(N, a, b, D, C=0.49):
    """
    Generates the time grid for solving PDEs numerically using a finite difference method.

    :param N: The number of spatial grid points.
    :param a: The left-end point of the spatial domain.
    :param b: The right-end point of the spatial domain.
    :param D: The diffusion coefficient.
    :param C: The Courant number, with a default value of 0.49.

    :returns: A tuple containing the time step size dt, the spatial step size dx, a numpy array t containing the time values at which to compute the solution, the number of time steps N_time, and a tuple containing the spatial domain interval (x_min, x_max).
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

def implicit_euler(N, gamma1, gamma2, a, b, D, N_time, x_int, dt, dx, IC):
    """
Solves the PDE numerically using the implicit Euler method.

    :param N: The number of spatial grid points.
    :param gamma1: The boundary condition value at x = 0.
    :param gamma2: The boundary condition value at x = 1.
    :param D: The diffusion coefficient.
    :param N_time: The number of time steps.
    :param x_int: A tuple containing the spatial domain interval.
    :param dt: The time step size.
    :param dx: The spatial step size.
    :param IC: A function representing the initial condition.

    :returns: A numpy array of shape (N_time+1, N-1) containing the numerical solution of the PDE.
    """
    # Compute C and build the required matrices
    C = D*dt/dx**2
    A_matrix = matrix_build(N, D)
    b_matrix = A_matrix @ dirichlet(N, gamma1, gamma2)
    identity = np.identity(N-1)
    # Initialize the solution array U with the IC
    U = np.zeros((N_time + 1, N-1))
    U[0,:] = IC(x_int, a, b)

    # Iterate over time steps using the implicit Euler method
    for i in range(0, N_time):
        U[i+1,:] = np.linalg.solve(identity - C*A_matrix, U[i,:] + C*b_matrix)
    
    return U

def crank(N, gamma1, gamma2, a, b, D, N_time, x_int, dt, dx, IC):
    """
    Solves the PDE numerically using the Crank-Nicolson method.

    :param N: The number of spatial grid points.
    :param gamma1: The boundary condition value at x = 0.
    :param gamma2: The boundary condition value at x = 1.
    :param D: The diffusion coefficient.
    :param N_time: The number of time steps.
    :param x_int: A tuple containing the spatial domain interval.
    :param dt: The time step size.
    :param dx: The spatial step size.
    :param IC: A function representing the initial condition.

    :returns: A numpy array of shape (N_time+1, N-1) containing the numerical solution of the PDE.
    """
    #Compute C and initialise the required matrices
    C = D*dt/dx**2
    A_matrix = matrix_build(N, D)
    b_matrix = A_matrix @ dirichlet(N, gamma1, gamma2)
    identity = np.identity(N-1)

    # Initialize the solution array U with the IC
    U = np.zeros((N_time + 1, N-1))
    U[0,:] = IC(x_int, a, b)

    # Iterate over time steps using the Crank-Nicolson method
    for i in range(0, N_time):
        U[i+1,:] = np.linalg.solve(identity - C/2 * A_matrix, (identity + C/2 * A_matrix) @ U[i,:] + C * b_matrix)
    
    return U

def linear_diffusion_sol(N, D, a, b, C=0.49):
    """
    Computes the true solution for a linear diffusion problem.

    :param N: The number of spatial grid points.
    :param D: The diffusion coefficient.
    :param a: The left endpoint of the spatial domain.
    :param b: The right endpoint of the spatial domain.
    :param t: A numpy array containing the time grid points.
    :param N_time: The number of time steps.
    :param x_int: A tuple containing the spatial domain interval.

    :returns: A numpy array of shape (N_time+1, N-1) containing the true solution of the linear diffusion problem.
    """
    dt, dx, t, N_time, x_int = time_grid(N, a, b, D, C)
    u_true = np.zeros((N_time+1, N-1))
    for n in range(0, N_time):
        u_true[n,:] = np.exp(-(b-a)**-2 *D*np.pi**2*t[n]) * np.sin(np.pi*(x_int-a)/(b-a))
    return u_true

def solve_PDE(method, N, D, gamma1, gamma2, a, b, IC, dt=None, C=0.49):
    """
    Solves a general PDE using the specified method.

    :param method: The method to use for solving the PDE, e.g., 'explicit_euler', 'implicit_euler', 'crank', 'RK4_method'.
    :param N: The number of spatial grid points.
    :param D: The diffusion coefficient.
    :param gamma1: The boundary condition value at x = 0.
    :param gamma2: The boundary condition value at x = 1.
    :param a: The left-end initial condition parameter.
    :param b: The right-end initial condition parameter.
    :param IC: A function representing the initial condition.
    :param t_final: The final time, default is 1.
    :param C: The Courant number, with a default value of 0.49.

    :returns: A numpy array containing the numerical solution of the PDE.
    """
    if dt is None:
        dt, dx, t, N_time, x_int = time_grid(N, a, b, D, C)
    else:
        _, dx, t, N_time, x_int = time_grid(N, a, b, D, C)
    
    if method == 'explicit_euler':
        return explicit_euler(N, D, gamma1, gamma2, a, b, dt, dx, t, N_time, x_int, IC), x_int
    elif method == 'implicit_euler':
        return implicit_euler(N, gamma1, gamma2, a, b, D, N_time, x_int, dt, dx, IC), x_int
    elif method == 'crank':
        return crank(N, gamma1, gamma2, a, b, D, N_time, x_int, dt, dx, IC), x_int
    elif method == 'RK4_method':
        return RK4_method(N, D, gamma1, gamma2, a, b, dt, dx, t, x_int, IC), x_int
    else:
        raise ValueError("Invalid method. Choose from 'explicit_euler', 'implicit_euler', 'crank', 'RK4_method'.")

def main():
    U_explicit, x_int = solve_PDE(method='explicit_euler', N=20, D=1, gamma1=0, gamma2=0, a=0, b=1, IC=linear_diffusion_IC1)
    U_RK4, x_int = solve_PDE(method='RK4_method', N=20, D=1, gamma1=0, gamma2=0, a=0, b=1, IC=linear_diffusion_IC1)
    u_true = linear_diffusion_sol(N=20, D=1, a=0, b=1)
    animate_solution(U_explicit, u_true, x_int)

    #Part 2
    U_implicit, x_int = solve_PDE(method='implicit_euler', N=100, D=0.1, gamma1=0, gamma2=0, a=0, b=1, dt=0.1, IC=linear_diffusion_IC2)
    U_crank, x_int = solve_PDE(method='crank', N=100, D=0.1, gamma1=0, gamma2=0, a=0, b=1, dt=0.1, IC=linear_diffusion_IC2)
    plt.plot(x_int, U_implicit[2,:], 'bx', label='implicit at t=2')
    plt.plot(x_int, U_crank[2,:], 'rx', label='crank at t=2')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u')
    plt.show()    

    #Part 3
    print('true sol = ', np.exp(-0.2*np.pi**2))
    y_interp1 = scipy.interpolate.interp1d(U_implicit[2,:], x_int, kind='linear')
    print('implicit euler = ', y_interp1(0.5))

    y_interp2 = scipy.interpolate.interp1d(U_crank[2,:], x_int, kind='linear')
    print('crank = ', y_interp2(0.5))

if __name__ == "__main__":
    main()