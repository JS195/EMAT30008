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

def source_function(N, integer):
    """
    Constructs a vector representing the source function of a one-dimensional 
    differential equation.
    
    :param N: Integer specifying the number of interior grid points in the domain.
    :param integer: Scalar specifying the constant value of the source function.
    
    :returns: Numpy ndarray of shape (N-1,) representing the source function of the differential 
    equation, with all elements equal to integer.
    """
    q = np.ones((N-1),)
    return q*integer

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

def BVP_solver(N, a, b, gamma1, gamma2, D, integer=1, source=None, boundary="dirichlet", k_a=None, k_b=None, h_a=None, h_b=None):
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
        if k_a is None or k_b is None or h_a is None or h_b is None:
            raise ValueError("k_a, k_b, h_a, and h_b must be provided for Robin boundary conditions.")
        b_matrix = np.zeros(N-1)
        A_matrix += robin(N, gamma1, gamma2, k_a, k_b, h_a, h_b, dx)

    if source:
        b_matrix -= ((dx)**2) * source_function(N, integer)

    x_ans = np.linalg.solve(A_matrix, b_matrix)
    return x_ans, x_int

def main():
    # Defining some variables
    N=20
    a=0
    b=1
    gamma1=0.0
    gamma2=1.0
    D = 1

    # Part 1
    x_ans, x_int = BVP_solver(N, a, b, gamma1, gamma2, D, source = False)
    u_true = true_sol(x_int,a,b,gamma1,gamma2, D)

    plt.plot(x_int, x_ans, 'o', label = "Numerical")
    plt.plot(x_int, u_true,'k',label='Exact')
    plt.legend()
    plt.show()

    # Part 2
    gamma2 = 0.0
    x_ans2, x_int2 = BVP_solver(N, a, b, gamma1, gamma2, D, source=source_function)
    u_ans2 = true_ans_part2(x_int2, a, b, gamma1, gamma2, D, 1)

    plt.plot(x_ans2, 'o', label = "Numerical")
    plt.plot(u_ans2,'k',label='Exact')
    plt.legend()
    plt.show()

    def solve_Bratu(N, a, b):
        grid = finite_grid(N, a, b)
        x = grid[0]
        dx = grid[1]
        x_int = grid[2]
        u_old = np.zeros(N-1,)
        for i in range(20):
            Bratu = np.exp(u_old*0.1)
            print(Bratu)
            A_matrix = matrix_build(N,D)
            b_matrix = - ((dx)**2)*source_function(N, Bratu) - A_matrix @ dirichlet(N,gamma1,gamma2)
            u_new = np.linalg.solve(A_matrix, b_matrix)
            u_ans = true_ans_part2(x_int, a, b, gamma1, gamma2, D, Bratu)
            u_old = u_new
        plt.plot(x[1:-1], u_new, 'o', label = "Numerical")
        plt.plot(x_int, u_ans,'k',label='Exact')
        plt.legend()
        plt.show()
    solve_Bratu(20, 0, 1)

if __name__ == "__main__":
    main()