import numpy as np
import matplotlib.pyplot as plt
import math
import time

def euler_step(f, x, t, dt):
    """
    :descript: Performs an euler step
    :param f: Function defining an ODE or ODE system
    :param x: Starting value of x
    :param t: Starting time value
    :param dt: Time step size
    :returns: The value of x after one timestep, and the new value of t
    """
    x_new = x + dt * f(x,t)
    t_new = t + dt
    return x_new, t_new

def RK4_step(f, x, t, dt):
    """
    :descript: Performs a step using the Runge-Kutta-4 method
    :param f: Function defining an ODE or ODE system
    :param x: Starting value of x
    :param t: Starting time value
    :param dt: Time step size
    :returns: The value of x after one timestep, and the new value of t
    """
    k1 = dt * f(x, t)
    k2 = dt * f(x + k1/2, t + dt/2)
    k3 = dt * f(x + k2/2, t + dt/2)
    k4 = dt * f(x + k3, t + dt)
    x_new = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    t_new = t + dt
    return x_new, t_new

def solve_to(f, x, t, t1, dt_max, solver='rk4'):
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
            x_new,t_new = euler_step(f, x, t, dt)
        elif solver == 'rk4':
            x_new,t_new = RK4_step(f, x, t, dt)
        x = x_new
        t = t_new
    return x_new, t_new

def solve_odes(f, x0, t0, t1, dt_max, solver='rk4'):
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
    n = int((t1 - t0) / dt_max)
    sol = np.zeros((n+1, len(x0)))
    sol[0] = x
    for i in range(n):
        dt = min(dt_max, t1 - t)
        if solver == 'euler':
            x, t = euler_step(f, x, t, dt)
        elif solver == 'rk4':
            x, t = RK4_step(f, x, t, dt)
        sol[i+1] = x
    return sol.T, np.linspace(t0, t1, n+1)


# Errors of the two methods plotted on the same graph
def errors(f, h, x0, t0, t1):
    """
    :descript: Creates a loglog graph of the error produced from the rk4 and euler method
    :param f: Function defining an ODE or ODE system
    :param h: The size of the timestep increase
    :param x0: Starting value of x
    :param t0: Starting time value
    :param t1: Final time value
    :returns: None
    """    
    errorsEuler = []
    errorsRK4 = []
    dt_max = h

    for i in range(1000):
        ans1 = solve_to(f, x0, t0, t1, dt_max, 'euler')[0]
        error1 = abs(math.e - ans1)
        errorsEuler.append(error1)
        ans2 = solve_to(f, x0, t0, t1, dt_max,'rk4')[0]
        error2 = abs(math.e - ans2)
        errorsRK4.append(error2)
        dt_max = dt_max + h

    tstep = np.linspace(h, dt_max, 1000)
    plt.loglog(tstep, errorsEuler,'r', label = "Euler")
    plt.loglog(tstep, errorsRK4, 'b', label = "RK4")
    plt.xlabel('log(tstep)')
    plt.ylabel('log(absolute error)')
    plt.title('Size of Timestep Against Error Produced from the RK4 and Euler Methods')
    plt.legend()
    plt.show()

#Time taken for the two methods over 1000 iterations
def timing(f, x0, t0, t1):
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
        ans1, t1 = solve_to(f, x0, t0, t1, 0.001, 'euler')
        n = n+1
    toc = time.perf_counter()
    tic1 = time.perf_counter()
    n=0
    while n<1000:
        ans2, t2 = solve_to(f, x0, t0, t1, 10,'rk4')
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

