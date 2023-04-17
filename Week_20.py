from Week_19_matrices import *
import numpy as np 
import matplotlib.pyplot as plt
from math import ceil
import matplotlib.animation as animation

C = 0.49
dt= C*dx**2/D
t_final = 1
N_time = int(np.ceil(t_final/dt))
t = dt*np.arange(N_time)

N=20
a=0
b=1
gamma1=0.0
gamma2=0.0
x=np.linspace(a,b,N+1)
dx=(b-a)/N
x_int=x[1:-1]
D = 1
integer = 1

A_matrix = matrix_build(N,D)
b_matrix = A_matrix @ boundary_conditions(N,gamma1,gamma2)

U = np.zeros((N_time + 1,N-1))
U[0,:] = np.sin(np.pi*(x_int-a)/(b-a))
    
for i in range(0,N_time-1):
    U[i+1,:] = U[i,:] + (dt*D/dx**2)*(A_matrix@U[i,:]+b_matrix) 

u_true = np.zeros((N_time+1, N-1))
for n in range(0, N_time):
    u_true[n,:] = np.exp(-(b-a)**2*D*np.pi**2*t[n]) * np.sin(np.pi*(x_int-a)/(b-a))

# Plot the true solution
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