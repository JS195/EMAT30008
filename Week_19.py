import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

N=20
a=0
b=1
alpha=0.0
beta=1.0
x=np.linspace(a,b,N+1)
dx=(b-a)/N
x_int=x[1:-1]

def dirichlet_problem(u, N, dx, alpha, beta):
    F = np.zeros(N-1)

    F[0] = (u[1] - 2*u[0] + alpha) / dx**2

    for i in range(1, N-2):
        F[i] = (u[i+1] - 2*u[i] + u[i-1]) / dx**2

    F[N-2] = (beta - 2*u[N-2] + u[N-3]) / dx**2

    return F

def true_ans(x,a,b,alpha,beta):
    answer = ((beta-alpha)/(b-a))*(x-a)+alpha
    return np.array(answer)


u_guess = 0.1*x_int

sol = root(dirichlet_problem, u_guess, args = (N,dx,alpha,beta))

print(sol.message)

u_int = sol.x
u_true = true_ans(x, a, b, alpha, beta)

#plt.plot(x_int, u_int, 'o', label = "Numerical")
#plt.plot(u_true, u_true,'k',label='Exact')
#plt.legend()
#plt.show()

######################################################################

def extendo_problem(u, N, dx, alpha, beta, f, D):
    F = np.zeros(N-1)

    F[0] = D*(u[1] - 2*u[0] + alpha) / dx**2 - f[0]

    for i in range(1, N-2):
        F[i] = D*(u[i+1] - 2*u[i] + u[i-1]) / dx**2 - f[i]

    F[N-2] = D*(beta - 2*u[N-2] + u[N-3]) / dx**2 - f[N-2]

    return F

def true_extendo(x,a,b,alpha,beta,D):
    answer = (-1/2*D)*(x-a)*(x-b) + ((beta-alpha)/(b-a))*(x-a)+alpha
    return np.array(answer)

D=2
u_guess = 0.1*x_int

f = np.zeros(len(x_int))

sol = root(extendo_problem, u_guess, args = (N,dx,alpha,beta,f,D))

print(sol.message)

u_int = sol.x
u_true = true_extendo(x_int, a, b, alpha, beta, D)

plt.plot(x_int, u_int, 'o', label = "Numerical")
plt.plot(x_int, u_true,'k',label='Exact')
plt.legend()
plt.show()