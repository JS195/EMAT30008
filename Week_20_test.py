#from Week_14 import 
from Week_19 import dirichlet, source, finite_grid, matrix_build
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

N=20
a=0
b=1
gamma1=0.0
gamma2=0.0
x=np.linspace(a,b,N+1)
dx=(b-a)/N
x_int=x[1:-1]
D = 2
integer = 1

D = 0.5

a = 0
b = 1
alpha = 0.0
beta = 0.0

f = lambda x: np.zeros(np.size(x))

N =20
x=np.linspace(a,b,N+1)
x_int = x[1:-1]
dx=(b-a)/N

C = 0.49

dt= C*dx**2/D
t_final =1
N_time = int(np.ceil(t_final/dt))
t = dt*np.arange(N_time)

grid = finite_grid(N, a, b)
x = grid[0]
dx = grid[1]
x_int = grid[2]

A_matrix = matrix_build(N,D)
b_matrix = - ((dx)**2)*source(N, integer) - A_matrix @ dirichlet(N,gamma1,gamma2)

def PDE(t, u, A_matrix, b_matrix):
    return D/dx**2 * (A_matrix @ u + b_matrix)

sol = solve_ivp(PDE, (0, t_final), f(x_int)) #= (D, A_matrix, b_matrix)

t = sol.t
u = sol.y

N_time = np.size(t)
print(N_time, 'time steps required')

plt.figure()
plt.plot(np.diff(t))
plt.xlabel(f'$n$')
plt.ylabel(f'$\Delta t$')
plt.tight_layout()
plt.show()