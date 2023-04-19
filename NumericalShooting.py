import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import fsolve
from scipy.signal import find_peaks
from TestingFunctions import predator_prey, pred_prey_pc
from ODEsolver import solve_odes, plot_different_parameters

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
    plt.plot(sol[:,0], sol[:,1])
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

def shooting(f, phase_cond):
    def G(u0T, pars):
        def F(u0, T):
            sol, t = solve_odes(f, x0=u0, t0=0, t1=T, dt_max=0.01, solver='rk4', pars=pars)
            final_sol = sol[-1, :]
            return final_sol
        T, u0 = u0T[-1], u0T[:-1]
        return np.append(u0 - F(u0, T), phase_cond(u0, pars=pars))
    return G

def find_shoot_orbit(f, phase_cond, u0T, pars):
    orbit = fsolve(shooting(f, phase_cond), u0T, pars)
    return orbit

def main():
    params = [[1.0, 0.1, 0.1], [1.0, 0.25, 0.1], [1.0, 0.4, 0.1]]
    plot_different_parameters(predator_prey, x0=[1,1], t0=0, t1=200, dt_max=0.01, params=params)

    # Isolate periodic orbit
    orbit = iso_orbit(predator_prey, x0=[1,1], t0=0, t1=500, dt_max=0.01, pars=[1.0,0.2,0.1])
    print(orbit)

    #Predator Prey shooting/ root finding
    pars = [1.0, 0.2, 0.1]
    u0T = [0.6, 0.3, 20]
    shooting_orbit = find_shoot_orbit(predator_prey, pred_prey_pc, u0T, pars)
    print(shooting_orbit)

    #Plotting phase portrait
    plot_phase_portrait(pars=pars)

if __name__ == "__main__":
    main()