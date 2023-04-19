import numpy as np

def euler_number(x, t):
    """
    Defines the ODE dx/dt = x.

    :param x: Value of x.
    :param t: Time t.

    :returns: Value of dx/dt = x.
    """
    return x

def true_euler_number(t):
    """
    Defines the true solution of dx/dt = x.

    :param t: Time t.

    :returns: Value of dx/dt = x at time t.
    """
    return np.exp(t)

def func2(x ,t):
    """
    Defines the second order ODE d_xx = -x.

    :param x: A vector of parameter values (x, y).
    :param t: Time value.

    :returns: An array of dx/dt = y and dy/dt = -x at (x, t).
    """
    X = x[0]
    y = x[1]
    dxdt = y
    dydt = -X
    return np.array([dxdt, dydt])

def predator_prey(X, t, pars):
    """
    Defines the predator-prey equations

    :param X: Vector of (x, y) values.
    :param t: Time value.
    :param pars: Other paramters required to define the equation (a, b, d).

    :returns: Array of derivatives dx/dt and dy/dt.
    """
    x = X[0]
    y = X[1]
    a, b, d = pars[0], pars[1], pars[2]
    dxdt = x * (1 - x) - (a * x * y) / (d + x)
    dydt = b * y * (1 - (y / x))
    return np.array([dxdt, dydt])

def pred_prey_pc(x0, pars):
    """
    Returns the predator-prey phase condition of dx/dt(0) = 0.

    :param x0: The initial condition for the ODE system.
    :param pars: Additional arguments to pass to the predator-prey funtion.

    :returns: The phase condition of the predator-prey system.
    """
    return predator_prey(x0, 0, pars)[0]

def hopf(U, t, pars):
    """
    Defines the predator-prey equations.

    :param X: Vector of (x, y) values.
    :param t: Time value.
    :param pars: Other paramters required to define the equation (a, b, d).

    :returns: Array of derivatives dx/dt and dy/dt.
    """
    u1 = U[0]
    u2 = U[1]
    beta, sigma = pars[0], pars[1]
    du1dt = beta * u1 -u2 + sigma*u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)
    return np.array([du1dt, du2dt])

def hopf_pc(x0, pars):
    """
    Returns the predator-prey phase condition of dx/dt(0) = 0.

    :param x0: The initial condition for the ODE system.
    :param *pars: Additional arguments to pass to the predator-prey funtion.

    :returns: The phase condition of the predator-prey system.
    """
    return hopf(x0, 0, pars)[0]