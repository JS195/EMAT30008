import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from NumericalShooting import find_shoot_orbit
from scipy.optimize import root
from Functions_and_ODEs import hopf_bif, hopf_bif_pc, cubic

def natural_continuation(f, #Funciton or ODE system
                         u0, #initial guess
                         min_par, #minimum par value
                         max_par, # maximum par value
                         no_steps, #number of steps 
                         phase_cond = 'None', #phase conditiion
                         discretisation = 'shooting'
                         ):
    sol_list = []
    par_list = np.linspace(min_par, max_par, no_steps)
    if phase_cond != 'None':
        if discretisation == 'shooting':
            for par in par_list:
                sol = find_shoot_orbit(f, phase_cond, u0, par) # no need to update parameters, good guess already
                sol_list.append(sol)
        return np.array(sol_list), par_list
    for par in par_list:
        sol = root(lambda x: f(x, par), u0)
        sol_list.append(sol.x)
        u0 = sol.x
    return np.array(sol_list), par_list




results, pars = natural_continuation(hopf_bif, [1.2, 1.0, 4], 0, 2, 30, hopf_bif_pc)
norm_np_sol_list = np.linalg.norm(results[:, :-1], axis = 1)
plt.plot(pars, norm_np_sol_list)
plt.xlabel('beta value')
plt.ylabel('||x||')
plt.show()

results, pars = natural_continuation(cubic, 0, -2, 2, 30)
plt.plot(pars,results)
plt.xlabel('c')
plt.ylabel('x')
plt.show()