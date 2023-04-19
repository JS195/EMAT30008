import numpy as np
import matplotlib.pyplot as plt
from Archive.OldODEsolver import solve_odes
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import math

def predator_prey(X, t, a=1.0, b=0.25, d=0.1):
    x = X[0]
    y = X[1]
    dxdt = x * (1 - x) - (a * x * y) / (d + x)
    dydt = b * y * (1 - (y / x))
    return np.array([dxdt, dydt])

def hopf_bif(X, t, b=0.5, s=1.0):
    u1 = X[0]
    u2 = X[1]
    du1dt = (b * u1) - u2 + ((s * u1) * (u1 ** 2 + u2 ** 2))
    du2dt = u1 + (b * u2) + ((s * u2) * (u1 ** 2 + u2 ** 2))
    return np.array([du1dt, du2dt])

def hopf_phase_condition(x0, b, s):
    return hopf_bif(x0, 0, b, s)[0]

b = 0.5
s = 1.0

# Behaviour in the long-time limit
X0 = [1, 1]
t = np.linspace(0, 20, 100)
sol, t = solve_odes(predator_prey, x0=[1,1], t0=0, t1=200, dt_max=0.1, solver='rk4')
plt.plot(t, sol[0], label='prey population')
plt.plot(t, sol[1], label='predator population')
plt.legend()
plt.show()

def hopf_actual(t, beta, theta):
    u1 = beta * math.cos(t + theta)
    u2 = beta * math.sin(t + theta)
    return u1, u2

def hopf_bif(X, t, beta):
    (u1, u2) = X
    du1dt = (beta * u1) - u2 + u1 * (u1 ** 2 + u2 ** 2) - u1 * ((u1 ** 2 + u2 ** 2) ** 2)
    du2dt = u1 + (beta * u2) + u2 * (u1 ** 2 + u2 ** 2) - u2 * ((u1 ** 2 + u2 ** 2) ** 2)
    return np.array([du1dt, du2dt])

def hopf_phase_condition(x0, beta):
    return hopf_bif(x0, 0, beta)[0]

def shooting(f):
    def G(u0T, beta):
        def F(u0, T):
            t_span = [1e-6, T]
            t_eval = np.linspace(1e-6, T, 100)
            sol = solve_ivp(fun=lambda t, X: f(X, t, beta), t_span=t_span, y0=[0.2,0.2], t_eval=t_eval, method='RK45')
            final_sol = sol.y[:,-1]
            return final_sol
        T, u0 = u0T[-1], u0T[:-1]
        return np.append(u0 - F(u0, T), hopf_phase_condition(u0, beta))
    return G

def find_shooting_orbit(f, u0T, beta):
    fsolve_sol = fsolve(shooting(f), u0T, (beta), full_output=True)
    shooting_orbit = fsolve_sol[0]
    return shooting_orbit

pred_prey_u0T = np.array([1.0,1.0,-2.0])
beta = 2.0
#Evaluate the actual Hopf bifurcation orbit
t_eval = np.linspace(0, 10*math.pi, 100)
actual_orbit = np.array([hopf_actual(t, beta, 0.0) for t in t_eval])

# Evaluate the shooting Hopf bifurcation orbit
found_shooting_orbit = find_shooting_orbit(hopf_bif, pred_prey_u0T, beta)
t_span = [1e-6, found_shooting_orbit[-1]]
t_eval = np.linspace(1e-6, found_shooting_orbit[-1], 100)
shooting_orbit = solve_ivp(fun=lambda t, X: hopf_bif(X, t, beta), t_span=t_span, y0=[0.2,0.2], t_eval=t_eval, method='RK45').y

# Compute the difference between the actual and shooting orbits
orbit_diff = actual_orbit - shooting_orbit.T
plt.plot(orbit_diff)
plt.show()
print("Orbit difference:")
print(orbit_diff)