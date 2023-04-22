import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import fsolve
from scipy.signal import find_peaks
from ExampleFunctions import predator_prey, pred_prey_pc, hopf, hopf_pc, three_dim_hopf, three_dim_hopf_pc
from ODEsolver import solve_odes, plot_different_parameters

def plot_phase_portrait(func=predator_prey,  x0=[1, 1], t0=0, t1=200, dt_max=0.01, solver='rk4', **kwargs):
    """
    Plots the phase portrait of an ODE or ODE system.
    
    :param func: The function defining the ODE or ODE system.
    :param x0: The initial condition for the ODE system.
    :param t0: The initial time for the ODE system.
    :param t1: The final time for the ODE system.
    :param dt_max: The maximum time step size allowed by the solver.
    :param solver: The ODE solver to use. Should be one of 'rk4' (default) or 'euler'.
    :param **kwargs: Optional. Additional keyword arguments to pass to the ODE solver.

    :returns: None, but produces a plot of the phase portrait.
    """
    sol, t = solve_odes(func, x0=x0, t0=t0, t1=t1, dt_max=dt_max, solver=solver, **kwargs)
    plt.plot(sol[:,0], sol[:,1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Predator-Prey Phase Portrait')
    plt.show()

def iso_orbit(f, x0, t0, t1, dt_max, **kwargs):
    """
    Finds the limit cycle initial conditions and time period for a system of ODEs using the Runge-Kutta 
    4th order solver.

    :param f: Function defining a system of ODEs.
    :param x0: Starting value of the dependent variable(s).
    :param t0: Starting time value.
    :param t1: Final time value.
    :param dt_max: Maximum step size.
    :param **kwargs: Optional. Any additional input keyword arguments.
    
    :returns: A list containing the initial conditons, and the time period of the limit cycle, if 
    one exists. If no limit cycle is found, returns None.
    """
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
    """
    Solves a boundary value problem using the shooting method.

    :param f: Function defining a system of ODEs.
    :param phase_cond: Function defining the boundary conditions to be satisfied.
    
    :returns: A function G that takes as input a list u0T representing the initial guess 
    for the solution, and a dictionary of parameter values, and returns the difference between 
    the actual boundary condition and the initial guess boundary condition.
    """
    def G(u0T, pars):
        """
        Defines a function G that is used by the shooting method to solve a boundary value problem.

        :param u0T: List representing the initial guess for the solution, where the last element 
        is the guess for the final time value, and the remaining elements are the guess for the 
        initial value(s).
        :param pars: Dictionary of parameter values.
        
        :returns: A numpy array representing the difference between the actual boundary condition 
        and the initial guess boundary condition. The first elements of the array correspond to the 
        difference between the actual and guessed initial value(s), while the last element corresponds 
        to the difference between the actual and guessed final time value.
        """
        def F(u0, T):
            """
            Solves a boundary value problem with shooting method.

            :param f: Function defining an ODE or ODE system.
            :param phase_cond: Function defining the boundary conditions.
            :param u0T: Array of initial guesses for the solution and the final time.
            :param pars: Any additional input parameters.
            
            :returns: Array of residuals between the boundary conditions and the solution obtained by 
            the shooting method.
            """
            sol, t = solve_odes(f, x0=u0, t0=0, t1=T, dt_max=0.01, solver='rk4', pars=pars)
            final_sol = sol[-1, :]
            return final_sol
        T, u0 = u0T[-1], u0T[:-1]
        return np.append(u0 - F(u0, T), phase_cond(u0, pars=pars))
    return G

def find_shoot_orbit(f, phase_cond, u0T, pars):
    """
    Finds the periodic orbit of a dynamical system using the shooting method.

    :param f: Function defining the dynamical system.
    :param phase_cond: Function defining the phase condition of the periodic orbit.
    :param u0T: Initial guess for the period and the initial condition of the orbit.
    :param pars: Parameters for the dynamical system.

    :returns: The initial conditions and the time period or the orbit as an array.
    """
    orbit = fsolve(shooting(f, phase_cond), u0T, pars)
    return orbit

def main():
    params = [[1.0, 0.1, 0.1], [1.0, 0.25, 0.1], [1.0, 0.4, 0.1]]
    plot_different_parameters(predator_prey, x0=[1,1], t0=0, t1=200, dt_max=0.01, params=params)
    
    #Plotting phase portrait
    pars = [1.0, 0.2, 0.1]
    plot_phase_portrait(pars=pars)

    # Isolate periodic orbit
    orbit = iso_orbit(predator_prey, x0=[1,1], t0=0, t1=500, dt_max=0.01, pars=[1.0,0.2,0.1])
    print('The true values of the predator prey orbit:', orbit)

    #Predator Prey shooting/ root finding
    # Using the true values from before to provide an initial guess
    u0T = [0.6, 0.3, 20]
    shooting_orbit = find_shoot_orbit(predator_prey, pred_prey_pc, u0T, pars)
    print('The shooting values of the predator prey orbit: ', shooting_orbit)

    #Testing the shooting code using the supercritical Hopf bifurcation
    pars = [0.3, -1]
    orbit = iso_orbit(hopf, [1,1], 0, 200, 0.01, pars=pars)
    print('The true values of the hopf orbit:', orbit)

    # Using the true values from before to provide an initial guess
    u0T = [0.6, 0.001, 6]
    shooting_orbit = find_shoot_orbit(hopf, hopf_pc, u0T, pars)
    print('The shooting values of the hopf orbit: ', shooting_orbit)

    #Testing the shooting code using the three dimensional hopf system
    pars = [0.3, -1]
    orbit = iso_orbit(three_dim_hopf, [1,1,1], 0, 200, 0.01, pars=pars)
    print('The true values of the three dim hopf orbit:', orbit)

    # Using the true values from before to provide an initial guess
    u0T = [0.6, 0.001, 0.001, 6]
    shooting_orbit = find_shoot_orbit(three_dim_hopf, three_dim_hopf_pc, u0T, pars)
    print('The shooting values of the three dim hopf orbit: ', shooting_orbit)


if __name__ == "__main__":
    main()