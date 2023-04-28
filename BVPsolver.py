import numpy as np
import matplotlib.pyplot as plt
from ExampleFunctions import true_sol, true_ans_part2

def matrix_build(N,D):
    """
    Constructs a (N-1)x(N-1) matrix representing a finite difference approximation to 
    the Laplacian operator on a regular grid.
    
    :param N: Number of grid points.
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
    
    :param N: Number of grid points.
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
    """
    Creates a matrix A that represents a Neumann boundary condition with 
    zero-flux at both ends of a 1D domain.

    :param N: Number of grid points.
    :param alpha: Scalar needed for calculating boundary values.
    :param beta: Scalar needed for calculating boundary values.
    :param dx: Grid spacing in the 1D domain.

    :returns: A (N-1)x(N-1) matrix that represents the Neumann boundary condition.
    """
    A = np.zeros((N-1, N-1))
    A[0, 0] = -1
    A[0, 1] = 1
    A[-1, -2] = -1
    A[-1, -1] = 1
    return A * alpha / dx

def robin(N, alpha, beta, dx, RobinArgs):
    """
    Creates a matrix A that represents a Robin boundary condition.

    :param N: Number of grid points.
    :param alpha: Scalar needed for calculating boundary values.
    :param beta: Scalar needed for calculating boundary values.
    :param dx: Grid spacing in the 1D domain.
    :param RobinArgs: A list containing the coefficients required for a Robin bc.

    :returns: A (N-1)x(N-1) matrix that represents the Robin boundary condition.
    """
    k_a, h_a, k_b, h_b = RobinArgs
    A = np.zeros((N-1, N-1))
    A[0, 0] = k_a + h_a * dx
    A[0, 1] = -k_a
    A[-1, -2] = -k_b
    A[-1, -1] = k_b + h_b * dx
    return A

def source_function(N, integer, x_int=0, u=0, x_dependant=False, u_dependant=False):
    """
    Creates a source function for a PDE that is either a constant, dependant on x, or 
    dependant on the solution of the PDE.
    
    :param N: Number of grid points.
    :param integer: Any constant integer.
    :param x_int: An array of the interior grid points. Defaults to dummy parameter 0.
    :param u: The value of the solution at each time step. Only required if 
        u_dependant=True. Defaults to dummy parameter 0.
    :x_dependant: True or False, if the source depends on x. Defaults to False.
    :u_dependant: True or False, if the source depends on the solution u. Defaults to False.

    :returns: An array corresponding to the source function.    
    """
    if x_dependant:
        q = x_int*integer
    else:
        q = np.ones((N-1),)*integer

    if u_dependant:
        q *= (u + integer)

    return q

def finite_grid(N, a, b):
    """
    Constructs a finite grid for solving a one-dimensional differential equation.
    
    :param N: Number of grid points.
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
    """
    Helper function to the BVP_solver. Solves a system of linear equations using.

    :param N: Number of grid points.
    :param A_matrix: An (N-1)x(N-1) ndarray that represents the matrix A in the linear system Ax=b.
    :param dx: Grid spacing in the 1D domain.
    :param x_int: An integer value that determines how the source function depends on x.
                  If x_dependant is True, then x_int is multiplied by the x-coordinate of each point.
                  Otherwise, the source function is a constant with a value of x_int.
    :param b_matrix: An (N-1) column vector that represents the vector b in the linear system Ax=b.
    :param x_dependant: A boolean that determines whether the source function depends on the x-coordinate.
    :param integer: A constant integer value.
    :param source: An optional parameter that represents the source function.

    :returns: A tuple (u, x_int) where u is an (N-1) column vector that represents the solution x of the linear system Ax=b
              and x_int is the integer value used to create the source function.
    """

    if source:
        b_matrix -= ((dx)**2) * source_function(N, integer, x_int, x_dependant)

    u = np.linalg.solve(A_matrix, b_matrix)
    return u, x_int

def Iterative_solver(N, A_matrix, dx, x_int, b_matrix, integer, source, x_dependant, u_dependant, tol, max_iter):
    """
    Helper function for BVP_solver. Solves a linear system of equations Ax=b iteratively using the Jacobi method,
    until the solution converges to a given tolerance or the maximum number of iterations is reached.

    :param N: Number of grid points.
    :param A_matrix: An (N-1)x(N-1) ndarray that represents the matrix A in the linear system Ax=b.
    :param dx: Grid spacing in the 1D domain.
    :param x_int: An integer value that determines how the source function depends on x.
                  If x_dependant is True, then x_int is multiplied by the x-coordinate of each point.
                  Otherwise, the source function is a constant with a value of x_int.
    :param b_matrix: An (N-1) column vector that represents the vector b in the linear system Ax=b.
    :param integer: A constant integer value.
    :param source: An optional parameter that represents the source function.
    :param x_dependant: A boolean that determines whether the source function depends on the x-coordinate.
    :param u_dependant: A boolean that determines whether the source function depends on the dependent variable u.
    :param tol: Tolerance for the convergence of the iterative solver.
    :param max_iter: Maximum number of iterations allowed for the iterative solver.

    :returns: A tuple (u, x_int) where u is an (N-1) column vector that represents the solution x of the linear system Ax=b
              and x_int is the integer value used to create the source function.
    """    
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
    """
    Solves a one-dimensional boundary value problem (BVP) using finite difference methods.
    The BVP is defined by the differential equation Du'' + gamma1*u' + gamma2*u = f(x), where u(0)=gamma1 and u(1)=gamma2.

    :param N: Number of grid points.
    :param a: Starting point of the domain.
    :param b: End point of the domain.
    :param gamma1: The boundary value u(0).
    :param gamma2: The boundary value u(1).
    :param D: Diffusion coefficient.
    :param integer: A constant integer value.
    :param source: An optional parameter that represents the source function.
    :param boundary: A string that represents the type of boundary condition.
                     Possible values: "dirichlet", "neumann", "robin".
    :param RobinArgs: A list that contains the values k_a, k_b, h_a, and h_b for the Robin boundary condition.
                      Required if boundary="robin".
    :param x_dependant: A boolean that determines whether the source function depends on the x-coordinate.
    :param u_dependant: A boolean that determines whether the dependent variable u is involved in the source function.
    :param tol: Tolerance for the convergence of the iterative solver.
    :param max_iter: Maximum number of iterations allowed for the iterative solver.

    :returns: A tuple (x_ans, x_int) where x_ans is an (N-1) column vector that represents the solution u of the BVP
              and x_int is the integer value used to create the source function.
    """    
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