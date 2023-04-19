import numpy as np

def euler_number(x ,t):
    """
    :descript: Defines the ODE dx/dt = x
    :param x: Value of x
    :param t: Time t
    :returns: Value of dx/dt = x
    """
    return x

def true_euler_number(t):
    return np.exp(t)

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

def predator_prey(X, t, pars):
    x = X[0]
    y = X[1]
    a, b, d = pars[0], pars[1], pars[2]
    dxdt = x * (1 - x) - (a * x * y) / (d + x)
    dydt = b * y * (1 - (y / x))
    return np.array([dxdt, dydt])