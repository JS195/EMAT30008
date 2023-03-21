import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

N=20
a=0
b=1
gamma1=0.0
gamma2=1.0
x=np.linspace(a,b,N+1)
dx=(b-a)/N
x_int=x[1:-1]

#def dirichlet_problem(u, N, dx, alpha, beta):
#    F = np.zeros(N-1)

#    F[0] = (u[1] - 2*u[0] + alpha) / dx**2

#    for i in range(1, N-2):
#        F[i] = (u[i+1] - 2*u[i] + u[i-1]) / dx**2

#    F[N-2] = (beta - 2*u[N-2] + u[N-3]) / dx**2

#    return F

#def true_ans(x,a,b,alpha,beta):
#    answer = ((beta-alpha)/(b-a))*(x-a)+alpha
#    return np.array(answer)


#u_guess = 0.1*x_int

#sol = root(dirichlet_problem, u_guess, args = (N,dx,gamma1,gamma2))

#print(sol.message)

#u_int = sol.x
#u_true = true_ans(x, a, b, alpha, beta)

#plt.plot(x_int, u_int, 'o', label = "Numerical")
#plt.plot(u_true, u_true,'k',label='Exact')
#plt.legend()
#plt.show()

######################################################################
def problem(u, N, dx, alpha, beta, f, D):
    F = np.zeros(N-1)

    F[0] = D*(u[1] - 2*u[0] + alpha) / dx**2 + f[0]

    for i in range(1, N-2):
        F[i] = D*(u[i+1] - 2*u[i] + u[i-1]) / dx**2 + f[i]

    F[N-2] = D*(beta - 2*u[N-2] + u[N-3]) / dx**2 + f[N-2]

    return F

def true_sol(x,a,b,alpha,beta, D):
    answer = (-1/2*D)*((x-a)*(x-b)) + ((beta-alpha)/(b-a))*(x-a)+alpha
    return np.array(answer)
 
D=1
u_guess = 0.1*x_int

def source_term(x):
    return 1

f = np.array([source_term(i*dx) for i in range(1, N)])

sol = root(problem, u_guess, args = (N,dx,gamma1,gamma2,f,D))
print(sol.message)
u_int = sol.x

u_true = true_sol(x_int, a, b, gamma1, gamma2, D)

plt.plot(x_int, u_int, 'o', label = "Numerical")
plt.plot(x_int, u_true,'k',label='Exact')
plt.legend()
plt.show()

#######################################################################################
def matrix_build(N,D):
    matrix = np.zeros(((N-1),(N-1)))
    np.fill_diagonal(matrix, -2*D)
    for i in range((N-1)):
        for j in range((N-1)):
            if i == j-1 or i==j+1:
                matrix[i][j]= D
    return matrix

def boundary_conditions(N, alpha, beta):
    vector = np.zeros((N-1),)
    vector[0] = alpha
    vector[-1] = beta
    return vector
    
def source(N, integer):
    q = np.ones((N-1),)
    q = q * integer
    return q

def true_ans(x,a,b,alpha,beta, D, integer):
    answer = (-integer)/(2*D)*((x-a)*(x-b)) + ((beta-alpha)/(b-a))*(x-a)+alpha
    return np.array(answer)

def finite_grid(N, a, b):
    x=np.linspace(a,b,N+1)
    dx=(b-a)/N
    x_int=x[1:-1]
    return x, dx, x_int

N=20
a=0
b=1
gamma1=0.0
gamma2=0.0
x=np.linspace(a,b,N+1)
dx=(b-a)/N
x_int=x[1:-1]
D = 2
integer = 7

grid = finite_grid(N, a, b)
x = grid[0]
dx = grid[1]
x_int = grid[2]
A_matrix = matrix_build(N,D)
b_matrix = - ((dx)**2)*source(N, integer) - A_matrix @ boundary_conditions(N,gamma1,gamma2)
x_ans = np.linalg.solve(A_matrix, b_matrix)
u_ans = true_ans(x_int, a, b, gamma1, gamma2, D, integer)

plt.plot(x_ans, 'o', label = "Numerical")
plt.plot(u_ans,'k',label='Exact')
plt.legend()
plt.show()
################################################################################################
def matrix_build(N,D):
    matrix = np.zeros(((N-1),(N-1)))
    np.fill_diagonal(matrix, -2*D)
    for i in range((N-1)):
        for j in range((N-1)):
            if i == j-1 or i==j+1:
                matrix[i][j]= D
    return matrix

def boundary_conditions(N, alpha, beta):
    vector = np.zeros((N-1),)
    vector[0] = alpha
    vector[-1] = beta
    return vector
    
def source(N, source_term):
    q = np.ones((N-1),)
    q = q * source_term
    return q

def true_ans(x,a,b,alpha,beta, D, source_term):
    answer = (-source_term)/(2*D)*((x-a)*(x-b)) + ((beta-alpha)/(b-a))*(x-a)+alpha
    return np.array(answer)

def finite_grid(N, a, b):
    x=np.linspace(a,b,N+1)
    dx=(b-a)/N
    x_int=x[1:-1]
    return x, dx, x_int

N=20
a=0
b=1
gamma1=0.0
gamma2=0.0
x=np.linspace(a,b,N+1)
dx=(b-a)/N
x_int=x[1:-1]
D = 1

grid = finite_grid(N, a, b)
x = grid[0]
dx = grid[1]
x_int = grid[2]

u_old = np.zeros(N-1,)
for i in range(20):
    Bratu = np.exp(u_old*0.1)
    A_matrix = matrix_build(N,D)
    b_matrix = - ((dx)**2)*source(N, Bratu) - A_matrix @ boundary_conditions(N,gamma1,gamma2)
    u_new = np.linalg.solve(A_matrix, b_matrix)
    u_ans = true_ans(x_int, a, b, gamma1, gamma2, D, Bratu)
    print(u_old)
    u_old = u_new


plt.plot(x[1:-1], u_new, 'o', label = "Numerical")
plt.plot(x_int, u_ans,'k',label='Exact')
plt.legend()
plt.show()
