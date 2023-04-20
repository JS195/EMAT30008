import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from NumericalShooting import shooting, find_shoot_orbit
from Functions_and_ODEs import hopf_bif, hopf_bif_pc

def cubic(x, pars):
    return x ** 3 - x + pars

def param_cont_cubic(func=cubic):
    sol_list = []
    u0 = 0
    for i in c:
        sol = fsolve(func, u0, args=(i,))
        sol_list.append(sol[0])
        u0 = sol[0]
    return np.array(sol_list)

#c = np.linspace(-2, 2, 100)
#solutions = param_cont_cubic(func=cubic)
#plt.plot(c,solutions)
#plt.xlabel('c')
#plt.ylabel('x')
#plt.show()

def param_cont_hopf(f=hopf_bif):
    beta_list = np.linspace(0, 2, 10)
    u0 = [1.2, 1.0, 4]
    sol_list = []
    for beta in beta_list:
        print(beta)
        sol = fsolve(shooting(hopf_bif, hopf_bif_pc), u0, (beta,))
        sol_list.append(sol)
    return np.array(sol_list), beta_list

results, par_list = param_cont_hopf()

print(results)

norm_np_sol_list = np.linalg.norm(results[:, :-1], axis = 1)

plt.xlabel('beta value')
plt.ylabel('||x||')
plt.show()