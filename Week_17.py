import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

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

c = np.linspace(-2, 2, 100)
solutions = param_cont_cubic(func=cubic)
#plt.plot(c, solutions)
#plt.xlabel('c')
#plt.ylabel('x')
#plt.show()



def hopf_bif(X, pars):
    (u1, u2) = X
    du1dt = (pars * u1) - u2 - (u1 * (u1**2 + u2**2))
    du2dt = u1 + (pars * u2) - (u2 * (u1**2 + u2 ** 2))
    return np.array([du1dt, du2dt])

def param_cont_hopf(func=hopf_bif):
    sol_list1 = []
    sol_list2 = []
    u0 = ([0,0])
    for i in beta:
        sol1, sol2 = fsolve(func, u0, args=(i,))
        sol_list1.append(sol1)
        sol_list2.append(sol2)
        u0 = ([sol1,sol2])
    return np.array(sol_list1), np.array(sol_list2)

beta = np.linspace(0, 2, 100)
sol_list1, sol_list2 = param_cont_hopf(func=hopf_bif)
plt.plot(sol_list1, sol_list2, beta)
plt.xlabel('sol_list1')
plt.ylabel('sol_list2')
plt.show()