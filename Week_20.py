import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Week_14 import RK4_step

D = 0.5

a = 0
b = 1
alpha = 0.0
beta = 0.0

f = lambda x: np.zeros(np.size(x))

N =20
x=np.linspace(a,b,N+1)
x_int = x[1:-1]
dx=(b-a)/N

C = 0.49

dt= C*dx**2/D
t_final =1
N_time = int(np.ceil(t_final/dt))
t = dt*np.arange(N_time)

print(N_time, 'time steps required')
print(f'The size of the time step is {dt:.5f}')

u = np.zeros((N_time+1, N-1))
u[0,:] = np.sin((np.pi*(x_int - a)) / (b - a))

for n in range(0, N_time):
    for i in range(0,N-1):
        if i ==0:
            u[n+1,0] = u[n,0]+C*(alpha-2 * u[n,0] + u[n,1])
        if 0 < i and i < N-2:
            u[n+1, i] = u[n,i] + C*(u[n,i+1] - 2 * u[n,i]+u[n,i-1])
        else:
            u[n+1,N-2] = u[n,N-2] +C*(beta-2*u[n,N-2] +u[n,N-3])


# Compute the true solution
u_true = np.zeros((N_time+1, N-1))
for n in range(0, N_time):
    u_true[n,:] = np.exp(-(b-a)**2*D*np.pi**2*t[n]) * np.sin(np.pi*(x_int-a)/(b-a))

# Plot the true solution
fig, ax = plt.subplots()
ax.set_ylim(0,1)
ax.set_xlabel(f'$x$')
ax.set_ylabel(f'$u(x,t)$')

line, = ax.plot(x_int, u_true[0, :], label='True solution')
line2, = ax.plot(x_int, u[0, :], 'ro', label='Numerical solution')

def animate(i):
    line.set_data((x_int, u_true[i,:]))
    line2.set_data((x_int, u[i,:]))
    return line, line2

ani = animation.FuncAnimation(fig, animate, frames=N_time, blit=False, interval=50)
ax.legend()
plt.show()

