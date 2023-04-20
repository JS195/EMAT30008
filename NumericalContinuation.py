import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from NumericalShooting import find_shoot_orbit
from scipy.optimize import root
from Functions_and_ODEs import hopf_bif, hopf_bif_pc, cubic

def natural_continuation(f, u0, min_par, max_par, no_steps, phase_cond = 'None', discretisation = 'shooting'):
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
    par_list = np.linspace(min_par, max_par, no_steps)
    if phase_cond != 'None':
        if discretisation == 'shooting':
            for par in par_list:
                sol = find_shoot_orbit(f, phase_cond, u0, par) # no need to update parameters, good guess already
                sol_list.append(sol)
        return np.array(sol_list), par_list
    for par in par_list:
        sol = fsolve(f, u0, args=(par,))
        sol_list.append(sol)
        u0 = sol # Ensures convergence
    return np.array(sol_list), par_list

def main():
    results, pars = natural_continuation(hopf_bif, [1.2, 1.0, 4], -1, 4, 30, hopf_bif_pc)
    norm_np_sol_list = np.linalg.norm(results[:, :-1], axis = 1)
    #plt.plot(pars, results[:,0], 'bx-')
    #plt.plot(pars,results[:,1],'rx-')
    plt.plot(pars, norm_np_sol_list, 'rx')
    plt.xlabel('beta value')
    plt.ylabel('||x||')
    plt.show()

    results, pars = natural_continuation(cubic, 0, -2, 2, 30)
    plt.plot(-pars, -results)
    plt.xlabel('c')
    plt.ylabel('x')
    plt.show()

if __name__ == "__main__":
    main()

