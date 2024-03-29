import numpy as np
import matplotlib.pyplot as plt
import math
import time
import warnings
from ExampleFunctions import euler_number, true_euler_number, predator_prey, func2

def euler_step(f, x, t, dt, **kwargs):
    """
    Performs a step using the Euler method

    :param f: Function defining an ODE or ODE system
    :param x: Starting value of x
    :param t: Starting time value
    :param dt: Time step size
    :param kwargs: Any additional input keyword arguments

    :returns: The value of x after one timestep, and the new value of t
    """
    x_new = x + dt * f(x, t, **kwargs)
    t_new = t + dt
    return x_new, t_new

def RK4_step(f, x, t, dt, **kwargs):
    """
    Performs a step using the Runge-Kutta-4 method

    :param f: Function defining an ODE or ODE system
    :param x: Starting value of x
    :param t: Starting time value
    :param dt: Time step size
    :param kwargs: Any additional input keyword arguments

    :returns: The value of x after one timestep, and the new value of t
    """
    k1 = dt * f(x, t, **kwargs)
    k2 = dt * f(x + k1/2, t + dt/2, **kwargs)
    k3 = dt * f(x + k2/2, t + dt/2, **kwargs)
    k4 = dt * f(x + k3, t + dt, **kwargs)
    x_new = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    t_new = t + dt
    return x_new, t_new

def solve_odes(f, x0, t0, t1, dt_max, solver='rk4', **kwargs):
    """
    Solves ODE f in the range t to t1 with initial condition x

    :param f: Function defining an ODE or ODE system
    :param x0: Starting value of x
    :param t0: Starting time value
    :param t1: Final time value
    :param dt_max: Maximum step size
    :solver: Defines the solver to use (either 'euler' or 'rk4')
    :param kwargs: Any additional input keyword arguments

    :returns: An array of x values at each time value 
    """
    # If incorrect inputs are specified
    if dt_max <= 0:
        raise ValueError("dt_max must be greater than 0")
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0")
    if solver not in ['euler', 'rk4']:
        raise ValueError("Invalid solver specified. Choose either 'euler' or 'rk4'")
    
    # Initialise variables
    t = t0
    x = np.array(x0)
    # Calculate the number of timesteps
    n = math.ceil((t1 - t0) / dt_max)
    sol = np.zeros((n+1, len(x0) if isinstance(x0, (list, tuple, np.ndarray)) else 1))
    sol[0] = x
    for i in range(n):
        #Calculate current step size
        dt = min(dt_max, t1 - t)
        #Select which method to use
        if solver == 'euler':
            x, t = euler_step(f, x, t, dt, **kwargs)
        elif solver == 'rk4':
            x, t = RK4_step(f, x, t, dt, **kwargs)
        #Store solution at current timestep
        sol[i+1] = x
        # Ignore runtime warnings as this only happens if t1 is large.
        warnings.filterwarnings("ignore", category=RuntimeWarning)
    return np.array(sol), np.linspace(t0, t1, n+1)

def error_difference(f, x0, t0, t1, true_solution, pars):
    """
    Calculates the difference in error between two ODE solvers (Euler and Runge-Kutta 4th order) 
    for a given ODE function and true solution, at different timestep values.

    :param f: Function defining an ODE or ODE system.
    :param x0: Starting value of x.
    :param t0: Starting time value.
    :param t1: Final time value.
    :param true_solution: Function defining the true solution to the ODE.
    :param pars: Parameters required to calculate the true solution.
    
    :returns: None (displays a plot of the error difference between the two solvers for different 
    timestep values).
    """
    rk4error = []
    eulererror = []
    # Array of timesteps to try
    timestep = np.logspace(-5, 3, 15)
    for dt in timestep:
        #Solve ODE at current timestep
        sol = solve_odes(f, x0, t0, t1, dt_max=dt, solver = 'rk4')[0]
        #Calculate absolute error
        error = abs(sol[-1] - true_solution(pars))
        rk4error.append(error)

        sol1 = solve_odes(f, x0, t0, t1, dt_max=dt, solver = 'euler')[0]
        error1 = abs(sol1[-1] - true_solution(pars))
        eulererror.append(error1)

    #Plot the output graph
    plt.loglog(timestep, rk4error, 'bx-', label='RK4 error')
    plt.loglog(timestep, eulererror, 'rx-', label='Euler error')
    plt.xlabel('Time step size')
    plt.ylabel('Absolute error')
    plt.legend()
    plt.show()

def plotter(x, y, linestyle, xlabel, ylabel, title, ax):
    """
    Plots the given data with specified labels and title on the provided axis.

    :param x: Array-like data for the x-axis.
    :param y: Array-like data for the y-axis.
    :param xlabel: String, label for the x-axis.
    :param ylabel: String, label for the y-axis.
    :param title: String, title for the plot.
    :param ax: Matplotlib axes object on which the plot will be drawn.

    :returns: None (modifies the provided axes object with the plot and labels).
    """
    ax.plot(x, y, linestyle)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def plot_different_parameters(f, x0, t0, t1, dt_max, params, solver='rk4'):
    """
    Plots the solution of a system of ODEs for different parameter values using a specified solver.

    :param f: Function defining a system of ODEs.
    :param x0: Starting value of the dependent variable(s).
    :param t0: Starting time value.
    :param t1: Final time value.
    :param dt_max: Maximum step size.
    :param params: List of dictionaries, where each dictionary contains the parameter values required 
    by the ODE system.
    :param solver: Defines the solver to use (either 'euler' or 'rk4').
    
    :returns: None (displays a plot of the solution of the ODE system for different parameter values).
    """
    fig, axs = plt.subplots(1, len(params), figsize=(12, 4))
    for i, p in enumerate(params):
        sol, t = solve_odes(f, x0, t0, t1, dt_max, solver, pars=p)
        plotter(t, sol[:, 0], '-', 'Time', 'Population', f"Prey - Parameters: {p}", axs[i])
        plotter(t, sol[:, 1], '-', 'Time', 'Population', f"Predator - Parameters: {p}", axs[i])
        
        axs[i].legend(['Prey', 'Predator'])
    plt.subplots_adjust(wspace=0.3)
    plt.show()

def main():
    pars = 1
    error_difference(euler_number, x0=1, t0=0, t1=1, true_solution = true_euler_number, pars = 1)

    start_timeEuler = time.perf_counter()
    ansEuler, tEuler = solve_odes(euler_number, x0=1, t0=0, t1=20, dt_max=0.01, solver = 'euler')
    end_timeEuler = time.perf_counter()

    start_timeRK4 = time.perf_counter()
    ansRK4, tRK4 = solve_odes(euler_number, x0=1, t0=0, t1=20, dt_max=1, solver = 'rk4')
    end_timeRK4 = time.perf_counter()

    print('Time taken for the Euler method to converge:', abs(end_timeEuler-start_timeEuler)) 
    print('Time taken for the RK4 method to converge:', abs(end_timeRK4-start_timeRK4))

    sol, t = solve_odes(func2, x0=[0.5,0.5], t0=0, t1=100, dt_max=1)
    plt.plot(t, sol)
    plt.xlabel('Time')
    plt.ylabel('Solution value')
    plt.show()

if __name__ == "__main__":
    main()