import numpy as np
import matplotlib.pyplot as plt
import math
import time

#PART 1
def euler_step(f,x,t,dt):
    step = x+dt*f(x,t)
    return step

def solve_to(f,x0,t0,t1,dt_max):
    t=[t0]
    x=[x0]
    while t[-1] < t1:
        dt = min(dt_max, t1 - t[-1])
        step = euler_step(f, x[-1], t[-1], dt)
        t.append(t[-1]+dt)
        x.append(step)
    return np.array(x), np.array(t)

# Calculating errors for a range of dt
def func(x, t): 
    return x

errors = []
x0 = 1
t0 = 0
t1 = 1
dt_max = 0.000001
for i in range(20):
    ans, t = solve_to(func, x0, t0, t1, dt_max)
    error = abs(math.e-ans[-1])
    errors.append(error)
    dt_max = dt_max*1.5
print(ans[-1],t[-1])

#Plotting the errors
tstep = np.linspace(0.000001, 1, 20)
plt.loglog(tstep, errors,'r')
plt.xlabel('log(tstep)')
plt.ylabel('log(absolute error)')
plt.title('A Loglog Graph of Timestep Size Against Error Produced from the Euler Method')
plt.show()

#PART 2
def RK4_step(f, x, t, h):
    k1 = h * f(x, t)
    k2 = h * f(x + h/2, t + k1/2)
    k3 = h * f(x + h/2, t + k2/2)
    k4 = h * f(x + h, t + k3)
    x_new = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    t_new = t + h
    return x_new, t_new

#Updated solve_to function
def solve_to(f,x0,t0,t1,h,solver=''):
    if solver == 'euler':
        t=t0
        x=x0
        while t < t1:
            dt = min(h, t1 - t)
            x = euler_step(f,x,t,dt)
            t = t + dt
        return x, t
    if solver == 'rk4':
        t = t0
        x = x0
        n = int((t1-t0)/h)
        for i in range(n):
             x, t = RK4_step(f, x, t, h)
        return x,t

def f(x, t):
    return x
t0 = 0
x0 = 1
h = 0.00001
x= solve_to(f, x0, t0, t1, h, 'rk4')
print(x)

# Plotting the error of the Euler method against the error of the RK4 method
errorsEuler = []
errorsRK4 = []
def func(x, t): 
    return x
x0 = 1
t0 = 0
t1 = 1
h = 0.000001
for i in range(35):
    ans, t = solve_to(func, x0, t0, t1, h, 'euler')
    error = abs(math.e-ans)
    errorsEuler.append(error)
    h = h*1.5
print(ans,t)

h = 0.000001
for i in range(35):
    ans, t = solve_to(func, x0, t0, t1, h,'rk4')
    error = abs(math.e-ans)
    errorsRK4.append(error)
    h = h*1.5
print(ans,t)

tstep = np.linspace(0.000001,0.0001,35)
plt.loglog(tstep, errorsEuler,'r', label = "Euler")
plt.loglog(tstep,errorsRK4, 'b', label = "RK4")
plt.xlabel('tstep')
plt.ylabel('absolute error')
plt.title('Size of Timestep Against Error Produced from the RK4 and Euler Methods')
plt.legend()
plt.show()

def func(x, t): 
    return x
x0 = 1
t0 = 0
t1 = 1
h=0.00001

start_timeEuler = time.perf_counter()
ansEuler, tEuler = solve_to(func, x0, t0, t1, h, 'euler')
end_timeEuler = time.perf_counter()

start_timeRK4 = time.perf_counter()
ansRK4, tRK4 = solve_to(func, x0, t0, t1, h,'rk4')
end_timeRK4 = time.perf_counter()

print(end_timeEuler, end_timeRK4)

from ODEsolver import errors 

errors(func, h, x0, t0, t1)