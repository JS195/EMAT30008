import numpy as np
import matplotlib.pyplot as plt
import math
import time
from TestingFunctions import predator_prey, pred_prey_pc, hopf, hopf_pc
from ODEsolver import solve_odes, plot_different_parameters
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

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

def iso_orbit(f, x0, t0, t1, dt_max, **kwargs):
    sol, t = solve_odes(f, x0, t0, t1, dt_max, 'rk4', **kwargs)
    x_coords = sol[:, 0]
    y_coords = sol[:, 1]
    peak_indices = find_peaks(x_coords)[0]
    previous_peak_index = None
    previous_peak_value = None
    for current_peak_index in peak_indices:
        if previous_peak_value is not None:
            if math.isclose(x_coords[current_peak_index], previous_peak_value, abs_tol=1e-4):
                period = t[current_peak_index] - t[previous_peak_index]
                orbit_info = [x_coords[current_peak_index], y_coords[current_peak_index], period]
                return orbit_info
        previous_peak_index = current_peak_index
        previous_peak_value = x_coords[current_peak_index]
    return None

def shooting(f):
    """
    :descript: Finds the initial conditions and time period of a periodic orbit for a given ODE system
    :param f: The function defining the ODE or ODE system
    :returns: A list or tuple containing the initial conditions of the periodic orbit and its period
    """
    def G(u0T, pars):
        """
        :descript: Helper function for shooting method. Returns the difference between the initial and final states 
        of the ODE system for a given set of initial conditions and time period
        :param u0T: List or tuple containing the initial conditions and time period of the periodic orbit
        :param pars: Additional parameters required to define the ODE system
        :returns: The difference between the final and initial states of the ODE system
        """
        def F(u0, T):
            """
            :descript: Helper function for shooting method. Solves the ODE system for a given set of initial conditions 
            and time period, and returns the final state.
            :param u0: Initial conditions of the ODE system
            :param T: Time period of the periodic orbit
            :returns: The final state of the ODE system
            """
            t_span = [1e-6, T]
            t_eval = np.linspace(1e-6, T, 100)
            sol = solve_ivp(fun=lambda t, X: f(X, t, pars), t_span=t_span, y0=u0, t_eval=t_eval, method='RK45')
            final_sol = sol.y[:,-1]
            return final_sol
        T, u0 = u0T[-1], u0T[:-1]
        return np.append(u0 - F(u0, T), hopf_pc(u0, pars))
    return G

def find_shooting_orbit(f, u0T, pars):
    """
    :descript: Finds the initial conditions and time period of a periodic orbit for a given ODE system
    :param f: The function defining the ODE or ODE system
    :param u0T: The initial conditions of the periodic orbit
    :param pars: Additional parameters for the ODE system
    :returns: A list or tuple containing the initial conditions of the periodic orbit and its period
    """
    fsolve_sol = fsolve(shooting(f), u0T, pars, full_output=True)
    shooting_orbit = fsolve_sol[0]
    return shooting_orbit

def main():
    params = [[1.0, 0.1, 0.1], [1.0, 0.25, 0.1], [1.0, 0.4, 0.1]]
    # plot_different_parameters(predator_prey, x0=[1,1], t0=0, t1=200, dt_max=0.01, params=params)

    # Isolate periodic orbit
    orbit = iso_orbit(predator_prey, x0=[1,1], t0=0, t1=500, dt_max=0.01, pars=[1.0,0.2,0.1])
    print(orbit)

    #Predator Prey shooting/ root finding
    pars = [1.0, 0.2, 0.1]
    pred_prey_u0T = np.array([0.6,0.3])
    found_shooting_orbit = find_shooting_orbit(predator_prey, pred_prey_u0T, pars)
    print(found_shooting_orbit)


if __name__ == "__main__":
    main()