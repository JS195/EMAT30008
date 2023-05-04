import numpy as np
import warnings
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from NumericalShooting import find_shoot_orbit
from ExampleFunctions import hopf_bif, standard_pc, cubic
from ODEsolver import plotter

def natural_continuation(f, u0, min_par, max_par, no_steps, phase_cond = None, discretisation = 'shooting'):
    """
    Computes and plots the natural continuation of a system of ODEs as a function of a parameter.

    :param f: The system of ODEs, given as a function of the form f(u, par), where u is the state vector and par is the parameter vector.
    :param u0: The initial guess for the solution of the ODE system.
    :param min_par: The minimum value of the parameter for which the continuation is computed.
    :param max_par: The maximum value of the parameter for which the continuation is computed.
    :param no_steps: The number of steps to take in the parameter space.
    :param phase_cond: Optional. If provided, it specifies the phase condition for the shooting method used to compute the continuation.
    :param discretisation: Optional. Specifies the discretization method to use. Can be 'shooting' (default) or 'fsolve'.
    
    :returns: A tuple containing two arrays: the first contains the solutions of the ODE system for each parameter value, and the second contains the corresponding parameter values.
    """
    sol_list = []
    #Generate list of parameter values
    par_list = np.linspace(min_par, max_par, no_steps)
    if phase_cond != None:
        if discretisation == 'shooting':
            for par in par_list:
                # find the periodic orbit for each parameter value
                try: 
                    sol = find_shoot_orbit(f, u0, par, phase_cond)
                # Catch exceptions and continue the loop.
                except Exception as e:
                    warnings.warn(f"Error encountered in find_shoot_orbit for par={par}: {e}")
                    continue
                sol_list.append(sol)
        return np.array(sol_list), par_list
    else: # No phase condition specified
        for par in par_list:
            # solve for the equilibrium solution for each parameter value
            try:
                sol = fsolve(f, u0, args=(par,))
            # Catch exceptions and continue the loop.
            except Exception as e:
                warnings.warn(f"Error encountered in fsolve for par={par}: {e}")
                continue
            sol_list.append(sol)
            u0 = sol # Ensures convergence
    return np.array(sol_list), par_list

def main():
    results, pars = natural_continuation(hopf_bif, [1.2, 1.0, 4], -1, 3, 6, standard_pc)
    fig, ax = plt.subplots()
    plotter(pars, results[:,0], 'bx', "Parameter Value", "Solution", "Natural Parameter Continuation of Hopf Bifurcation Equation", ax)
    plotter(pars, results[:,1], 'rx-', "Parameter Value", "Solution", "Natural Parameter Continuation of Hopf Bifurcation Equation", ax)
    plt.show()

    results, pars = natural_continuation(cubic, 0, -2, 2, 30)
    fig1, ax1 = plt.subplots()
    plotter(pars, results, '-', "Parameter Value", "Solution", "Natural Parameter Continuation of Cubic Equation", ax1)
    plt.show()

if __name__ == "__main__":
    main()