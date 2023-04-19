import numpy as np
import matplotlib.pyplot as plt
import math
import time
from TestingFunctions import euler_number, true_euler_number, predator_prey, func2

def euler_step(f, x, t, dt, **kwargs):
    """
    :descript: Performs an euler step
    :param f: Function defining an ODE or ODE system
    :param x: Starting value of x
    :param t: Starting time value
    :param dt: Time step size
    :returns: The value of x after one timestep, and the new value of t
    """
    x_new = x + dt * f(x, t, **kwargs)
    t_new = t + dt
    return x_new, t_new

def RK4_step(f, x, t, dt, **kwargs):
    """
    :descript: Performs a step using the Runge-Kutta-4 method
    :param f: Function defining an ODE or ODE system
    :param x: Starting value of x
    :param t: Starting time value
    :param dt: Time step size
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
    :descript: Solves ODE f in the range t to t1 with initial condition x
    :param f: Function defining an ODE or ODE system
    :param x0: Starting value of x
    :param t0: Starting time value
    :param t1: Final time value
    :param dt_max: Maximum step size
    :solver: Defines the solver to use (either 'euler' or 'rk4')
    :returns: An array of x values at each time value 
    """
    t = t0
    x = np.array(x0)
    n = math.ceil((t1 - t0) / dt_max)
    sol = np.zeros((n+1, len(x0) if isinstance(x0, (list, tuple, np.ndarray)) else 1))
    sol[0] = x
    for i in range(n):
        dt = min(dt_max, t1 - t)
        if solver == 'euler':
            x, t = euler_step(f, x, t, dt, **kwargs)
        elif solver == 'rk4':
            x, t = RK4_step(f, x, t, dt, **kwargs)
        sol[i+1] = x
    return np.array(sol), np.linspace(t0, t1, n+1)

# Errors of the two methods plotted on the same graph
def error_difference(f, x0, t0, t1, true_solution, pars):
    rk4error = []
    eulererror = []
    timestep = np.logspace(-5, 2, 15)
    for dt in timestep:
        sol, t = solve_odes(f, x0, t0, t1, dt_max=dt, solver = 'rk4')
        error = abs(sol[-1] - true_solution(pars))
        rk4error.append(error)

        sol1, t2 = solve_odes(f, x0, t0, t1, dt_max=dt, solver = 'euler')
        error1 = abs(sol1[-1] - true_solution(pars))
        eulererror.append(error1)

    plt.loglog(timestep, rk4error, 'x-')
    plt.loglog(timestep, eulererror, 'x-')
    plt.show()

def plot_different_parameters(f, x0, t0, t1, dt_max, params, solver='rk4'):
    fig, axs = plt.subplots(1, len(params), figsize=(12, 4))
    for i, p in enumerate(params):
        sol, t = solve_odes(f, x0, t0, t1, dt_max, solver, pars=p)
        axs[i].plot(t, sol[:, 0], label='Prey')
        axs[i].plot(t, sol[:, 1], label='Predator')
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Population')
        axs[i].set_title('Parameters: {}'.format(p))
        axs[i].legend()

    plt.subplots_adjust(wspace=0.3)
    plt.show()

def main():
    pars = 1
    error_difference(euler_number, x0=1, t0=0, t1=1, true_solution = true_euler_number, pars = pars)

    sol, t = solve_odes(func2, x0=[1,1], t0=0, t1=20, dt_max=0.01)
    plt.plot(sol)
    plt.show()

if __name__ == "__main__":
    main()