def implicit_euler(a, b, gamma1, gamma2, N, D, C, IC):
    dt, dx, t, N_time, x_int = time_grid(a, b, N, D, C)
    C = 0.1/dx**2
    A_matrix = matrix_build(N, D)
    b_matrix = A_matrix @ boundary_conditions(N, gamma1, gamma2)
    identity = np.identity(N-1)
    U = np.zeros((N_time + 1, N-1))
    U[0,:] = IC(x_int, a, b)

    for i in range(0, N_time-1):
        U[i+1,:] = np.linalg.solve(identity + C*A_matrix, U[i,:] + C*b_matrix)
    
    return U, N_time, x_int

def crank(a, b, gamma1, gamma2, N, D, C, IC):
    dt, dx, t, N_time, x_int = time_grid(a, b, N, D, C)
    C = 0.1/dx**2
    A_matrix = matrix_build(N, D)
    b_matrix = A_matrix @ boundary_conditions(N, gamma1, gamma2)
    identity = np.identity(N-1)
    U = np.zeros((N_time + 1, N-1))
    U[0,:] = IC(x_int, a, b)
    for i in range(0, N_time-1):
        U[i+1,:] = np.linalg.solve(identity - C/2 * A_matrix, (identity + C/2 * A_matrix) @ U[i,:] + C * b_matrix
    return U, N_time, x_int


































def linear_diffusion_IC(x_values, a, b):
    return np.sin(np.pi*(x_values-a)/(b-a))

def linear_diffusion_true_sol(t, n, x_int, a, b, D):
    return np.exp(-(b-a)**2*D*np.pi**2*t[n]) * np.sin(np.pi*(x_int-a)/(b-a))

U, N_time, x_int = explicit_euler(a=0, b=1, gamma1 = 0.0, gamma2=0.0, N=20, D=1, C=0.49, IC=linear_diffusion_IC)

u_true, N_time, x_int = true_sol=true_solution(a=0, b=1, N=20, D=1, C=0.49, true_sol=linear_diffusion_true_sol)


