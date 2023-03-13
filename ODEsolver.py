import numpy as np
import matplotlib.pyplot as plt
import math
import time

#Euler step
def euler_step(f,x,t,dt):
    x_new = x+dt*f(x,t)
    t_new = t+dt
    return x_new,t_new

#RK4 step
def RK4_step(f, x, t, h):
    k1 = h * f(x, t)
    k2 = h * f(x + k1/2, t + h/2)
    k3 = h * f(x + k2/2, t + h/2)
    k4 = h * f(x + k3, t + h)
    x_new = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    t_new = t + h
    return x_new, t_new

#Solve taking input as either 'Euler' or 'RK4'
def solve_to(f,x0,t0,t1,h,solver=''):
    t=t0
    x=x0
    while t < t1:
        dt = min(h, t1 - t)
        if solver == 'euler':
            x_new,t_new = euler_step(f,x,t,dt)
        elif solver == 'rk4':
            x_new,t_new = RK4_step(f, x, t, dt)
        x = x_new
        t = t_new
    return x_new, t_new

# Solve a system of ODEs taking input as either 'Euler' or 'RK4'
def solve_odes(f, x0, t0, t1, h, solver='euler'):
    t = t0
    x = np.array(x0)
    n = int((t1 - t0) / h)
    sol = np.zeros((n+1, len(x0)))
    sol[0] = x
    for i in range(n):
        dt = min(h, t1 - t)
        if solver == 'euler':
            x, t = euler_step(f, x, t, dt)
        elif solver == 'rk4':
            x, t = RK4_step(f, x, t, dt)
        sol[i+1] = x
    return sol.T, np.linspace(t0, t1, n+1)


# Errors of the two methods plotted on the same graph
def errors(f, h, x0, t0, t1):
    errorsEuler = []
    errorsRK4 = []
    d=h

    for i in range(1000):
        ans1, t1 = solve_to(f, x0, t0, t1, d, 'euler')
        error1 = abs(math.e-ans1)
        errorsEuler.append(error1)
        ans2, t2 = solve_to(f, x0, t0, t1, d,'rk4')
        error2 = abs(math.e-ans2)
        errorsRK4.append(error2)
        d = d+h

    tstep = np.linspace(h,d,1000)
    plt.loglog(tstep, errorsEuler,'r', label = "Euler")
    plt.loglog(tstep, errorsRK4, 'b', label = "RK4")
    plt.xlabel('log(tstep)')
    plt.ylabel('log(absolute error)')
    plt.title('Size of Timestep Against Error Produced from the RK4 and Euler Methods')
    plt.legend()
    plt.show()

#Time taken for the two methods over 1000 iterations
def timing(f, x0, t0, t1):
    tic = time.perf_counter()
    n=0
    while n<1000:
        ans1, t1 = solve_to(f, x0, t0, t1, 0.001, 'euler')
        n = n+1
    toc = time.perf_counter()
    tic1 = time.perf_counter()
    n=0
    while n<1000:
        ans2, t2 = solve_to(f, x0, t0, t1, 10,'rk4')
        n=n+1
    toc1 = time.perf_counter()
    return(toc-tic, toc1-tic1)