import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def cubic(x, c):
    return x ** 3 - x + c

c = np.linspace(-2, 2, 100)

def param_cont(func=cubic):
    sol_list = []
    u0 = 0
    for i in c:
        sol = fsolve(func, u0, args=(i,))
        sol_list.append(sol[0])
        u0 = sol[0]
    return np.array(sol_list)

solutions = param_cont(func=cubic)
plt.plot(c, solutions)
plt.xlabel('c')
plt.ylabel('x')
plt.show()
