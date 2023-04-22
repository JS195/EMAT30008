import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.interpolate
from BVPsolver import matrix_build, boundary_conditions, finite_grid
from ODEsolver import RK4_step
from ExampleFunctions import linear_diffusion_IC1, linear_diffusion_IC2

def true_sol_func(N, D, a, b, t, N_time, x_int):
    u_true = np.zeros((N_time+1, N-1))
    for n in range(0, N_time):
        u_true[n,:] = np.exp(-(b-a)**2*D*np.pi**2*t[n]) * np.sin(np.pi*(x_int-a)/(b-a))
    return u_true

def explicit_euler(N, D, gamma1, gamma2, a, b, dt, dx, t, N_time, x_int):
    A_matrix = matrix_build(N,D)
    b_matrix = A_matrix @ boundary_conditions(N,gamma1,gamma2)

    U = np.zeros((N_time + 1, N-1))
    U[0,:] = linear_diffusion_IC1(x_int, a, b)
    
    for i in range(0,N_time-1):
        U[i+1,:] = U[i,:] + (dt*D/dx**2)*(A_matrix@U[i,:]+b_matrix) 

    return U

def heat_equation_RK4(N, D, gamma1, gamma2, a, b, dt, dx, t, x_int):
    A_matrix = matrix_build(N,D)
    b_matrix = A_matrix @ boundary_conditions(N,gamma1,gamma2)

    U = np.zeros((len(t), N-1))
    U[0,:] = np.sin(np.pi*(x_int-a)/(b-a))
    
    def f(U, t):
        return D/dx**2 * (A_matrix @ U + b_matrix)
    
    for i in range(len(t)-1):
        U[i+1,:], _ = RK4_step(f, U[i,:], t[i], dt)
    
    return U

def animate_solution(U, u_true, x_int, N_time):
    fig, ax = plt.subplots()
    ax.set_ylim(0,1)
    ax.set_xlabel(f'$x$')
    ax.set_ylabel(f'$u(x,t)$')

    line, = ax.plot(x_int, u_true[0, :], label='True solution')
    line2, = ax.plot(x_int, U[0, :], 'ro', label='Numerical solution')

    def animate(i):
        line.set_data((x_int, u_true[i,:]))
        line2.set_data((x_int, U[i,:]))
        return line, line2

    ani = animation.FuncAnimation(fig, animate, frames=N_time, blit=False, interval=50)
    ax.legend()
    plt.show()

def time_grid(N, a, b, D, C=0.49):
    # C must be smaller than 0.5, else the system bugs out
    grid = finite_grid(N, a, b)
    dx = grid[1]
    x_int = grid[2]
    dt = C*dx**2/D
    t_final = 1
    N_time = int(np.ceil(t_final/dt))
    t = dt*np.arange(N_time)
    return dt, dx, t, N_time, x_int

def implicit_euler(N, gamma1, gamma2, D, N_time, x_int, dt, dx, IC):
    C = D*dt/dx**2
    A_matrix = matrix_build(N, D)
    b_matrix = A_matrix @ boundary_conditions(N, gamma1, gamma2)
    identity = np.identity(N-1)
    U = np.zeros((N_time + 1, N-1))
    U[0,:] = IC(x_int)

    for i in range(0, N_time-1):
        U[i+1,:] = np.linalg.solve(identity - C*A_matrix, U[i,:] + C*b_matrix)
    
    return U

def crank(N, gamma1, gamma2, D, N_time, x_int, dt, dx, IC):
    C = D*dt/dx**2
    A_matrix = matrix_build(N, D)
    b_matrix = A_matrix @ boundary_conditions(N, gamma1, gamma2)
    identity = np.identity(N-1)
    U = np.zeros((N_time + 1, N-1))
    U[0,:] = IC(x_int)

    for i in range(0, N_time-1):
        U[i+1,:] = np.linalg.solve(identity - C/2 * A_matrix, (identity + C/2 * A_matrix) @ U[i,:] + C * b_matrix)
    
    return U

def main():
    N=20
    a=0
    b=1
    gamma1=0.0
    gamma2=0.0
    D = 1

    dt, dx, t, N_time, x_int = time_grid(N, a, b, D)
    U_explicit = explicit_euler(N, D, gamma1, gamma2, a, b, dt, dx, t, N_time, x_int)
    U_RK4 = heat_equation_RK4(N, D, gamma1, gamma2, a, b, dt, dx, t, x_int)
    u_true = true_sol_func(N, D, a, b, t, N_time, x_int)
    animate_solution(U_explicit, u_true, x_int, N_time)

    #Part 2
    N=100
    D=0.1
    dt, dx, t, N_time, x_int = time_grid(N, a, b, D)
    U_implicit = implicit_euler(N, gamma1, gamma2, D, N_time, x_int, 0.1, dx, linear_diffusion_IC2)
    
    U_crank = crank(N, gamma1, gamma2, D, N_time, x_int, 0.1, dx, linear_diffusion_IC2)

    plt.plot(x_int, U_implicit[2,:], 'bx', label='implicit at t=2')
    plt.plot(x_int, U_crank[2,:], 'rx', label='crank at t=2')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u')
    plt.show()    

    #Part 3
    print('true sol = ', np.exp(-0.2*np.pi**2))
    y_interp1 = scipy.interpolate.interp1d(U_implicit[2,:], x_int, kind='linear')
    print('implicit euler = ', y_interp1(0.5))

    y_interp2 = scipy.interpolate.interp1d(U_crank[2,:], x_int, kind='linear')
    print('crank = ', y_interp2(0.5))

if __name__ == "__main__":
    main()