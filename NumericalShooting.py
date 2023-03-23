import numpy as np
import matplotlib.pyplot as plt
from ODEsolver import solve_odes
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

def predator_prey(X, t, pars):
    """
    :descript: Defines the predator-prey equations

    :param X: Vector of (x, y) values
    :param t: Time value
    :param pars: Other paramters required to define the equation (a, b, d)

    :returns: Array of derivatives dx/dt and dy/dt
    """
    x = X[0]
    y = X[1]
    a, b, d = pars[0], pars[1], pars[2]
    dxdt = x * (1 - x) - (a * x * y) / (d + x)
    dydt = b * y * (1 - (y / x))
    return np.array([dxdt, dydt])

def long_term_behaviour():
    """
    :descript: Produces subplots of the predator-prey function for different input parameters

    :returns: None
    """
    parameter_sets = [
        [1.0, 0.1, 0.1],
        [1.0, 0.25, 0.1],
        [1.0, 0.5, 0.1]
    ]
    num_plots = len(parameter_sets)
    for i in range(num_plots):
        pars = parameter_sets[i]
        sol, t = solve_odes(predator_prey, x0=[1,1], t0=0, t1=200, dt_max=0.01, solver='rk4', pars=pars)
        plt.subplot(num_plots, 1, i+1)
        plt.plot(t, sol[0], label='Prey')
        plt.plot(t, sol[1], label='Predator')
        plt.xlabel('Time')
        plt.ylabel('Pop, b = {}'.format(pars[1]))
        plt.legend()
    plt.show()

def plot_phase_portrait(func=predator_prey,  x0=[1, 1], t0=0, t1=200, dt_max=0.01, solver='rk4', **kwargs):
    """
    :descript: Plots the phase portrait of an ODE or ODE system

    :param func: The function defining the ODE or ODE system
    :param x0: The initial condition for the ODE system
    :param t0: The initial time for the ODE system
    :param t1: The final time for the ODE system
    :param dt_max: The maximum time step size allowed by the solver
    :param solver: The ODE solver to use. Should be one of 'rk4' (default) or 'euler'
    :param **kwargs: Additional keyword arguments to pass to the ODE solver

    :returns: None
    """
    sol, t = solve_odes(func, x0=x0, t0=t0, t1=t1, dt_max=dt_max, solver=solver, **kwargs)
    plt.plot(sol[0], sol[1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Predator-Prey Phase Portrait')
    plt.show()

def isolate_orbit(func=predator_prey,  x0=[1, 1], t0=0, t1=200, dt_max=0.01, solver='rk4', **kwargs):
    """
    :descript: Given an ODE system, function finds a periodic orbit and returns its initial conditions and time period

    :param func: The function defining the ODE or ODE system
    :param x0: The initial condition for the ODE system
    :param t0: The initial time for the ODE system
    :param t1: The final time for the ODE system
    :param dt_max: The maximum time step size allowed by the solver
    :param solver: The ODE solver to use. Should be one of 'rk4' (default) or 'euler'
    :param **kwargs: Additional keyword arguments to pass to the ODE solver

    :returns: A list or tuple containing the initial conditions of the periodic orbit and its period
    :raises RuntimeError: If no periodic orbit is found
    """
    sol, t = solve_odes(func, x0=x0, t0=t0, t1=t1, dt_max=dt_max, solver=solver, **kwargs)
    x_values = np.asarray(sol[0])
    max_indices = [i for i in find_peaks(x_values)[0]]
    previous_value = False
    previous_time = 0
    for i in max_indices:
        if previous_value:
            if np.isclose(x_values[i], previous_value, rtol=0, atol=1e-4):
                period = t[i] - previous_time
                orbit = [sol[0][i], sol[1][i], period]
                return orbit
        previous_value = x_values[i]
        previous_time = t[i]
    raise RuntimeError("No orbit found")

def phase_condition(x0, *pars):
    return predator_prey(x0, 0, *pars)[0]

def shooting(f):
    def G(u0T, pars):
        def F(u0, T):
            t_span = [1e-6, T]
            t_eval = np.linspace(1e-6, T, 100)
            sol = solve_ivp(fun=lambda t, X: f(X, t, pars), t_span=t_span, y0=u0, t_eval=t_eval, method='RK45')
            final_sol = sol.y[:,-1]
            return final_sol
        T, u0 = u0T[-1], u0T[:-1]
        return np.append(u0 - F(u0, T), phase_condition(u0, pars))
    return G

pars = [1.0, 0.5, 0.1]

def find_shooting_orbit(f, u0T, pars):
    fsolve_sol = fsolve(shooting(f), u0T, pars, full_output=True)
    shooting_orbit = fsolve_sol[0]
    return shooting_orbit

pred_prey_u0T = np.array([0.2,0.2,13])
found_shooting_orbit = find_shooting_orbit(predator_prey, pred_prey_u0T, pars)
print(found_shooting_orbit)