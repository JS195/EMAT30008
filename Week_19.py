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

def boundary_conditions(N, alpha, beta):
    vector = np.zeros((N-1),)
    vector[0] = alpha
    vector[-1] = beta
    return vector

def source_function(N, integer):
    q = np.ones((N-1),)
    return q*integer

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

def BVP_solver(N, a, b, gamma1, gamma2, D, integer = 1, source = False):
    grid = finite_grid(N, a, b)
    x = grid[0]
    dx = grid[1]
    x_int = grid[2]
    A_matrix = matrix_build(N,D)
    if source:
        b_matrix = - ((dx)**2)*source(N, integer) - A_matrix @ boundary_conditions(N,gamma1,gamma2)
    else:
        b_matrix = - boundary_conditions(N,gamma1,gamma2)
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
            b_matrix = - ((dx)**2)*source_function(N, Bratu) - A_matrix @ boundary_conditions(N,gamma1,gamma2)
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