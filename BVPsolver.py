import numpy as np
import matplotlib.pyplot as plt

def matrix_build(N,D):
    """
    Constructs a (N-1)x(N-1) matrix representing a finite difference approximation to 
    the Laplacian operator on a regular grid.
    
    :param N: Number of grid points along each dimension (excluding the boundary).
    :param D: Diffusion constant.
    
    :returns: Numpy ndarray of shape (N-1, N-1) representing the Laplacian operator on the grid, 
    multiplied by D.
    """
    matrix = np.zeros(((N-1),(N-1)))
    np.fill_diagonal(matrix, -2)
    for i in range((N-1)):
        for j in range((N-1)):
            if i == j-1 or i==j+1:
                matrix[i][j]= 1
    return matrix*D

def dirichlet(N, alpha, beta):
    """
    Constructs a vector representing the boundary conditions of a one-dimensional 
    differential equation.
    
    :param N: Integer specifying the number of interior grid points in the domain.
    :param alpha: Scalar specifying the value of the solution at the left boundary.
    :param beta: Scalar specifying the value of the solution at the right boundary.
    
    :return: Numpy ndarray of shape (N-1,) representing the boundary conditions of the differential 
    equation, with the first element equal to alpha and the last element equal to beta.
    """
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

def robin(N, gamma1, gamma2, k_a, k_b, h_a, h_b, dx):
    A = np.zeros((N-1, N-1))
    A[0, 0] = k_a + h_a * dx
    A[0, 1] = -k_a
    A[-1, -2] = -k_b
    A[-1, -1] = k_b + h_b * dx
    return A

def source_function(N, integer, x_int, u, x_dependent, u_dependent):
    if x_dependent:
        q = x_int*integer
    else:
        q = np.ones((N-1),)*integer

    if u_dependent:
        q *= (u + integer)

    return q

def true_sol(x,a,b,alpha,beta, D):
    answer = ((beta - alpha)/(b - a))*(x - a) + alpha
    return np.array(answer)

def true_ans_part2(x,a,b,alpha,beta, D, integer):
    answer = (-integer)/(2*D)*((x-a)*(x-b)) + ((beta-alpha)/(b-a))*(x-a)+alpha
    return np.array(answer)

def finite_grid(N, a, b):
    """
    Constructs a finite grid for solving a one-dimensional differential equation.
    
    :param N: Integer specifying the number of interior grid points in the domain.
    :param a: Scalar specifying the left boundary of the domain.
    :param b: Scalar specifying the right boundary of the domain.
    
    :returns: A tuple containing the following elements:
        - x: Numpy ndarray of shape (N+1,) representing the grid points, including the boundary points.
        - dx: Scalar representing the grid spacing.
        - x_int: Numpy ndarray of shape (N-1,) representing the interior grid points, i.e., the grid 
          points excluding the boundary points.
    """
    x=np.linspace(a,b,N+1)
    dx=(b-a)/N
    x_int=x[1:-1]
    return x, dx, x_int

def Matrix_solver(N, A_matrix, dx, x_int, b_matrix, x_dependant=False, integer=1, source=None):
    if source:
        b_matrix -= ((dx)**2) * source_function(N, integer, x_int, 0, x_dependant, u_dependent=False)

    u = np.linalg.solve(A_matrix, b_matrix)
    return u, x_int

def Iterative_solver(N, A_matrix, dx, x_int, b_matrix, integer, source, x_dependant, u_dependant, tol, max_iter):
    u = np.zeros(N-1)
    u_prev = np.ones(N-1) * np.inf
    iter_count = 0

    while np.linalg.norm(u - u_prev) > tol and iter_count < max_iter:
        u_prev = u.copy()

        if source:
            b_matrix_updated = b_matrix - ((dx)**2) * source_function(N, integer, x_int, u, x_dependant, u_dependant)
            u = np.linalg.solve(A_matrix, b_matrix_updated)
        else:
            u = np.linalg.solve(A_matrix, b_matrix)

        iter_count += 1
    if iter_count == max_iter:
        print("Warning: Maximum number of iterations reached. Solution may not have converged.")
    return(u, x_int)

def BVP_solver(N, a, b, gamma1, gamma2, D, integer=1, source=None, boundary="dirichlet", RobinArgs=None, x_dependant=False, u_dependant=False, tol=1e-6, max_iter=100):
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

    if u_dependant == False:
        x_ans, x_int = Matrix_solver(N, A_matrix, dx, x_int, b_matrix, x_dependant, integer, source)

    else:
        x_ans, x_int = Iterative_solver(N, A_matrix, dx, x_int, b_matrix, integer, source, x_dependant, u_dependant, tol, max_iter)

    return x_ans, x_int

def main():
    x_ans, x_int = BVP_solver(N=50, a=0, b=1, gamma1=0, gamma2=1, D=1, source=False)
    u_true = true_sol(x_int, a=0, b=1, alpha=0, beta=1, D=1)
    plt.plot(x_int, x_ans, 'o', label="Numerical")
    plt.plot(x_int, u_true, 'k', label='Exact')
    plt.legend()
    plt.show()

    # Part 2
    x_ans2, x_int2 = BVP_solver(N=50, a=0, b=1, gamma1=0, gamma2=0, D=1, integer=1, source=True, boundary="dirichlet", x_dependant=False)
    u_ans2 = true_ans_part2(x_int2, a=0, b=1, alpha=0, beta=0, D=1, integer=1)
    plt.plot(x_int2, x_ans2, 'o', label="Numerical")
    plt.plot(x_int2, u_ans2, 'k', label='Exact')
    plt.legend()
    plt.show()

    # Bratu Equation
    x_ansbratu, x_intbratu = BVP_solver(N=50, a=0, b=1, gamma1=0, gamma2=0, D=3, integer=0.1, source=True, boundary="dirichlet", x_dependant=True, u_dependant=True, tol=1e-6, max_iter=100)
    plt.plot(x_intbratu, x_ansbratu, 'o', label="Numerical")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()