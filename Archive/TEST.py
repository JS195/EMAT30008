import numpy as np
import matplotlib.pyplot as plt

def matrix_build(N,D):
    matrix = np.zeros(((N-1),(N-1)))
    np.fill_diagonal(matrix, -2)
    for i in range((N-1)):
        for j in range((N-1)):
            if i == j-1 or i==j+1:
                matrix[i][j]= 1
    return matrix*D

def dirichlet(N, alpha, beta):
    vector = np.zeros((N-1),)
    vector[0] = alpha
    vector[-1] = beta
    return vector

def neumann(N, alpha, beta, dx):
    A = np.zeros((N-1, N-1))
    A[0, 0] = -1
    A[0, 1] = 1
    A[-1, -2] = -1
    A[-1, -1] = 1
    return A * alpha / dx

def robin(N, gamma1, gamma2, dx, RobinArgs):
    k_a, k_b, h_a, h_b = RobinArgs
    A = np.zeros((N-1, N-1))
    A[0, 0] = k_a + h_a * dx
    A[0, 1] = -k_a
    A[-1, -2] = -k_b
    A[-1, -1] = k_b + h_b * dx
    return A

def source_function(N, integer, x_int, u, mu, x_dependent, u_dependent):
    if x_dependent:
        q = integer * x_int
    else:
        q = np.ones((N-1),) * integer

    if u_dependent:
        q *= (u + mu)

    return q


def true_sol(x,a,b,alpha,beta, D):
    answer = ((beta - alpha)/(b - a))*(x - a) + alpha
    return np.array(answer)

def true_ans_part2(x,a,b,alpha,beta, D, integer):
    answer = (-integer)/(2*D)*((x-a)*(x-b)) + ((beta-alpha)/(b-a))*(x-a)+alpha
    return np.array(answer)

def finite_grid(N, a, b):
    x=np.linspace(a,b,N+1)
    dx=(b-a)/N
    x_int=x[1:-1]
    return x, dx, x_int

def BVP_solver(N, a, b, gamma1, gamma2, D, integer=1, mu=1, source=None, boundary="dirichlet", RobinArgs=None, x_dependant=False, u_dependant=False, tol=1e-6, max_iter=100):
    grid = finite_grid(N, a, b)
    x = grid[0]
    dx = grid[1]
    x_int = grid[2]
    A_matrix = matrix_build(N, D)

    if boundary == "dirichlet":
        b_matrix = -dirichlet(N, gamma1, gamma2)
    elif boundary == "neumann":
        b_matrix = np.zeros(N-1)
        A_matrix += neumann(N, gamma1, gamma2, dx)
    elif boundary == "robin":
        if RobinArgs is None:
            raise ValueError("k_a, k_b, h_a, and h_b must be provided for Robin boundary conditions.")
        b_matrix = np.zeros(N-1)
        A_matrix += robin(N, gamma1, gamma2, dx, RobinArgs)

    u = np.zeros(N-1)
    u_prev = np.ones(N-1) * np.inf
    iter_count = 0

    while np.linalg.norm(u - u_prev) > tol and iter_count < max_iter:
        u_prev = u.copy()

        if source:
            b_matrix_updated = b_matrix - ((dx)**2) * source_function(N, integer, x_int, u, mu, x_dependant, u_dependant)
            u = np.linalg.solve(A_matrix, b_matrix_updated)
        else:
            u = np.linalg.solve(A_matrix, b_matrix)
        
        iter_count += 1

    if iter_count == max_iter:
        print("Warning: Maximum number of iterations reached. Solution may not have converged.")

    return u, x_int


N = 50
a = 0
b = 1
gamma1 = 0.0
gamma2 = 1.0
D = 1

# Part 1
x_ans, x_int = BVP_solver(N, a, b, gamma1, gamma2, D, source=False)
u_true = true_sol(x_int, a, b, gamma1, gamma2, D)

plt.plot(x_int, x_ans, 'o', label="Numerical")
plt.plot(x_int, u_true, 'k', label='Exact')
plt.legend()
plt.show()

# Part 2
gamma2 = 0.0
x_ans2, x_int2 = BVP_solver(N, a, b, gamma1, gamma2, D, source=source_function)
u_ans2 = true_ans_part2(x_int2, a, b, gamma1, gamma2, D, 1)

plt.plot(x_int2, x_ans2, 'o', label="Numerical")
plt.plot(x_int2, u_ans2, 'k', label='Exact')
plt.legend()
plt.show()


x_ansbratu, x_intbratu = BVP_solver(N, a, b, gamma1, gamma2, D, mu=0.1, source=True, boundary="dirichlet", x_dependant=False, u_dependant=True, tol=1e-6, max_iter=100)

plt.plot(x_intbratu, x_ansbratu, 'o', label="Numerical")

plt.legend()
plt.show()