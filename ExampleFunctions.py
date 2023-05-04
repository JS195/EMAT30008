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

def three_dim_hopf(U, t, pars):
    """
    Returns the time derivative of a 3D predator-prey system at a given time.

    :param U: A numpy array of length 3 containing the state variables (u1, u2, u3).
    :param t: The time.
    :param pars: A tuple containing the system parameters (beta, sigma).

    :returns: A numpy array of length 3 containing the time derivatives of the state variables.
    """
    u1, u2, u3 = U[0], U[1], U[2]
    beta, sigma = pars[0], pars[1]
    du1dt = beta * u1 -u2 + sigma*u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)
    du3dt = -u3
    return np.array([du1dt, du2dt, du3dt])

def hopf_bif(X, t, pars):
    """
    Returns the predator-prey phase condition of dx/dt(0) = 0 for a 3D system.

    :param x0: The initial condition for the ODE system.
    :param pars: A tuple containing the system parameters (beta, sigma).

    :returns: The phase condition of the predator-prey system.
    """
    (u1, u2) = X
    du1dt = (pars * u1) - u2 - (u1 * (u1**2 + u2**2))
    du2dt = u1 + (pars * u2) - (u2 * (u1**2 + u2 ** 2))
    return np.array([du1dt, du2dt])

def cubic(x, pars):
    """
    Computes a cubic polynomial with given parameters.

    :param x: The input variable.
    :param pars: A tuple containing the coefficients of the cubic polynomial.

    :returns: The value of the cubic polynomial at x.
    """
    a = 1
    b = 0
    c = -1
    return a * x**3 + b * x**2 + c * x + pars

def linear_diffusion_IC1(x_values, a, b):
    """
    Computes the initial condition for a linear diffusion problem.

    :param x_values: A numpy array containing the spatial grid points.
    :param a: The left endpoint of the spatial domain.
    :param b: The right endpoint of the spatial domain.

    :returns: A numpy array of the same shape as x_values containing the initial condition.
    """
    return np.sin(np.pi*(x_values-a)/(b-a))

def linear_diffusion_IC2(x_values):
    """
    Computes the initial condition for a linear diffusion problem.

    :param x_values: A numpy array containing the spatial grid points.

    :returns: A numpy array of the same shape as x_values containing the initial condition.
    """
    return np.sin(np.pi*(x_values))

def linear_diffusion_true_sol(t, n, x_int, a, b, D):
    """
    Computes the true solution for a linear diffusion problem.

    :param t: The time at which to evaluate the solution.
    :param n: The index of the time step to evaluate.
    :param x_int: A tuple containing the spatial domain interval.
    :param a: The left endpoint of the spatial domain.
    :param b: The right endpoint of the spatial domain.
    :param D: The diffusion coefficient.

    :returns: A numpy array of the same shape as x_int containing the true solution at time t[n].
    """
    return np.exp(-(b-a)**2*D*np.pi**2*t[n]) * np.sin(np.pi*(x_int-a)/(b-a))

def true_sol(x,a,b,alpha,beta, D):
    """
    Computes the true solution for a linear ODE problem.

    :param x: The input variable.
    :param a: The left endpoint of the spatial domain.
    :param b: The right endpoint of the spatial domain.
    :param alpha: The boundary condition value at x = a.
    :param beta: The boundary condition value at x = b.
    :param D: The diffusion coefficient.

    :returns: A numpy array of the same shape as x containing the true solution.
    """
    answer = ((beta - alpha)/(b - a))*(x - a) + alpha
    return np.array(answer)

def BVP_true_answer(x,a,b,alpha,beta, D, integer):
    """
    Computes the true solution for a linear ODE problem.

    :param x: The input variable.
    :param a: The left endpoint of the spatial domain.
    :param b: The right endpoint of the spatial domain.
    :param alpha: The boundary condition value at x = a.
    :param beta: The boundary condition value at x = b.
    :param D: The diffusion coefficient.
    :param integer: A constant integer value.

    :returns: A numpy array of the same shape as x containing the true solution.
    """    
    answer = (-integer)/(2*D)*((x-a)*(x-b)) + ((beta-alpha)/(b-a))*(x-a)+alpha
    return np.array(answer)

def true_sol_func(N, D, a, b, t, N_time, x_int):
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
    u_true = np.zeros((N_time+1, N-1))
    for n in range(0, N_time):
        u_true[n,:] = np.exp(-(b-a)**2*D*np.pi**2*t[n]) * np.sin(np.pi*(x_int-a)/(b-a))
    return u_true

def standard_pc(f, x0, pars):
    """
    Returns the phase condition dx/dt(0) = 0.

    :param f: The function.
    :param x0: The initial condition for the ODE system.
    :param *pars: Additional arguments to pass to the function.

    :returns: The phase condition of the system.
    """
    return f(x0, 0, pars)[0]