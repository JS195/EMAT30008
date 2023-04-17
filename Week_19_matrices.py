import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def matrix_build(N,D):
    matrix = np.zeros(((N-1),(N-1)))
    np.fill_diagonal(matrix, -2)
    for i in range((N-1)):
        for j in range((N-1)):
            if i == j-1 or i==j+1:
                matrix[i][j]= 1
    return matrix*D

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
integer = 2

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