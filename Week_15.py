import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.optimize import fsolve
from ODEsolver import solve_odes
from scipy.signal import find_peaks

# Predator-prey function
def predator_prey(X, t, pars):
    x = X[0]
    y = X[1]
    a, b, d = pars[0], pars[1], pars[2]
    dxdt = x * (1 - x) - (a * x * y) / (d + x)
    dydt = b * y * (1 - (y / x))
    return np.array([dxdt, dydt])

pars = [1.0, 0.25, 0.1]
# Behaviour in the long-time limit
t = np.linspace(0, 200, 1000)
sol, t = solve_odes(predator_prey, x0=[1,1], t0=0, t1=200, dt_max=0.01, solver='rk4', pars=pars)
plt.plot(t, sol[0], label='prey population')
plt.plot(t, sol[1], label='predator population')
plt.legend()
plt.show()

# Plot the phase portrait
plt.plot(sol[0], sol[1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Predator-Prey Phase Portrait')
plt.show()

#def nullcline(ode, u0range, index=0, points=500):
#    Vval = np.linspace(min(u0range), max(u0range), points)
#    Nval = np.zeros(np.size(Vval))
#    for (i, V) in enumerate(Vval):
#        result = root(lambda N: ode((V, N), np.nan, a, b, d)[index], 0)
#        if result.success:
#            Nval[i] = result.x
#        else:
#            Nval[i] = np.nan
#    return (Vval, Nval)

# V nullcline
#(Vval, Nval) = nullcline(predator_prey, (-20, 20), index=0)
#plt.plot(Vval, Nval, "g-")

# N nullcline
#(Vval, Nval) = nullcline(predator_prey, (-20, 20), index=1)
#plt.plot(Vval, Nval, "r-")
#plt.show()

def phase_condition(x0, a, b, d):
    return predator_prey(x0, 0, a,b,d)[0]

def shooting(f):
    def G(u0T, a, b,d):
        def F(u0, T):
            t_span = [1e-6, T]
            t_eval = np.linspace(1e-6, T, 100)
            sol = solve_ivp(fun=lambda t, X: f(X, t, a, b, d), t_span=t_span, y0=u0, t_eval=t_eval, method='RK45')
            final_sol = sol.y[:,-1]
            return final_sol
        T, u0 = u0T[-1], u0T[:-1]
        print(T)
        return np.append(u0 - F(u0, T), phase_condition(u0, a, b, d))
    return G

pars = [1.0, 0.5, 0.1]

def find_shooting_orbit(f, u0T, pars):
    a,b,d = pars[0], pars[1], pars[2]
    fsolve_sol = fsolve(shooting(f), u0T, (a,b,d), full_output=True)
    shooting_orbit = fsolve_sol[0]
    return shooting_orbit

pred_prey_u0T = np.array([0.2,0.2,20])
found_shooting_orbit = find_shooting_orbit(predator_prey, pred_prey_u0T, pars)
plt.plot(found_shooting_orbit)
plt.show()

print(found_shooting_orbit)

def isolate_orbit(func=predator_prey,  x0=[1, 1], t0=0, t1=200, dt_max=0.01, solver='rk4', **kwargs):
    solution, t = solve_odes(predator_prey, x0=x0, t0=t0, t1=t1, dt_max=dt_max, solver=solver, **kwargs)
    x_values = np.asarray(solution[0])
    max_indices = [i for i in find_peaks(x_values)[0]]
    previous_value = False
    previous_time = 0
    for i in max_indices:
        if previous_value:
            if np.isclose(x_values[i], previous_value, rtol=0, atol=1e-4):
                period = t[i] - previous_time
                orbit = [solution[0][i], solution[1][i], period]
                return orbit
        previous_value = x_values[i]
        previous_time = t[i]
    raise RuntimeError("No orbit found")
solution_array = isolate_orbit(pars=[1.0, 0.5, 0.1])
print('Initial values for the orbit found:', solution_array[:-1])
print('Period of the orbit found:', solution_array[-1])