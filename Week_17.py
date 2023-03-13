import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def cubic(x, c):
    return x ** 3 - x + c

def param_cont(func=cubic):
    sol_list=[]
    u0 = 0
    for c in range(-2,2):
        sol = np.array(fsolve(func, u0, args=c))
        sol_list.append(sol)
        u0=sol
    return np.array(sol_list)

solutions=param_cont(func=cubic)
plt.plot(solutions)
plt.show()