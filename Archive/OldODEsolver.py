import numpy as np
import matplotlib.pyplot as plt
import math
import time

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

def solve_to(f, x, t, t1, dt_max, solver='rk4', **kwargs):
    """
    :descript: Solves ODE f in the range t to t1 with initial condition x
    :param f: Function defining an ODE or ODE system
    :param x: Starting value of x
    :param t: Starting time value
    :param t1: Final time value
    :param dt_max: Maximum step size
    :solver: Defines the solver to use (either 'euler' or 'rk4')
    :returns: The value of x at the final time t1. 
    """
    while t < t1:
        dt = min(dt_max, t1 - t)
        if solver == 'euler':
            x_new,t_new = euler_step(f, x, t, dt, **kwargs)
        elif solver == 'rk4':
            x_new,t_new = RK4_step(f, x, t, dt, **kwargs)
        x = x_new
        t = t_new
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
    sol = np.zeros((n+1, len(x0) if isinstance(x0, (list, tuple)) else 1))
    sol[0] = x
    for i in range(n):
        dt = min(dt_max, t1 - t)
        if solver == 'euler':
            x, t = euler_step(f, x, t, dt, **kwargs)
        elif solver == 'rk4':
            x, t = RK4_step(f, x, t, dt, **kwargs)
        sol[i+1] = x
    if sol.shape[1] > 2:
        sol = sol.T
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

def true_euler_number(t):
    return np.exp(t)

#Time taken for the two methods over 1000 iterations
def timing(f, x0, t0, t1, *pars):
    """
    :descript: Times the two methods over 1000 runs
    :param f: Function defining an ODE or ODE system
    :param x0: Starting value of x
    :param t0: Starting time value
    :param t1: Final time value
    :returns: The time taken for the euler step and rk4 method 
    """ 
    tic = time.perf_counter()
    n=0
    while n<1000:
        ans1, t1 = solve_to(f, x0, t0, t1, 0.001, 'euler', *pars)
        n = n+1
    toc = time.perf_counter()
    tic1 = time.perf_counter()
    n=0
    while n<1000:
        ans2, t2 = solve_to(f, x0, t0, t1, 10,'rk4', *pars)
        n=n+1
    toc1 = time.perf_counter()
    return(toc-tic, toc1-tic1)

def func1(x ,t):
    """
    :descript: Defines the ODE dx/dt = x
    :param x: Value of x
    :param t: Time t
    :returns: Value of dx/dt = x
    """
    return x

def func2(x ,t):
    """
    :descript: Defines the second order ODE d_xx = -x
    :param x: A vector of parameter values (x, y)
    :param t: Time value
    :returns: An array of dx/dt = y and dy/dt = -x at (x, t) 
    """
    X = x[0]
    y = x[1]
    dxdt = y
    dydt = -X
    return np.array([dxdt, dydt])

