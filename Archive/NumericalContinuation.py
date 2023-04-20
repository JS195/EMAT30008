import numpy as np
from scipy.optimize import fsolve

def RK4_step(f, x, t, dt, *pars):
    k1 = dt * f(x, t, *pars)
    k2 = dt * f(x + k1/2, t + dt/2, *pars)
    k3 = dt * f(x + k2/2, t + dt/2, *pars)
    k4 = dt * f(x + k3, t + dt, *pars)
    x_new = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    t_new = t + dt
    return x_new, t_new

def solve_odes(f, x0, t0, t1, dt_max, solver='rk4', *pars):
    t = t0
    x = np.array(x0)
    n = int((t1 - t0) / dt_max)
    sol = np.zeros((n+1, len(x0)))
    sol[0] = x
    for i in range(n):
        dt = min(dt_max, t1 - t)
        if solver == 'euler':
            x, t = euler_step(f, x, t, dt, *pars)
        elif solver == 'rk4':
            x, t = RK4_step(f, x, t, dt, *pars)
        sol[i+1] = x
    return sol.T, np.linspace(t0, t1, n+1)

def hopf_phase_condition(u0, pars):
    return hopf_bif(u0, pars)[0]

def hopf_bif(X, pars):
    u1, u2 = X[0], X[1]
    beta = pars[0]
    du1dt = (beta * u1) - u2 - (u1 * (u1**2 + u2**2))
    du2dt = u1 + (beta * u2) - (u2 * (u1**2 + u2 ** 2))
    return (du1dt, du2dt)

def shooting(f):
    def G(u0T, phase_cond, *pars):
        def F(u0, T):
            t0 = 0.1
            t1 = T
            dt_max = 0.01
            sol = solve_odes(f, u0, t0, t1, dt_max, pars)
            final_sol = sol[0][:,-1]
            return final_sol
        T = u0T[-1]
        u0T = u0
        return np.append(u0 - F(u0, T), phase_cond(u0, *pars))
    return G

vary_par = (0, 2)
min_par = vary_par[0]
max_par = vary_par[1]
parameter_array = np.linspace(min_par, max_par, 20)
u0 = np.array([1.0, 6.0])
sol_list = []
pars = [0]
initial_pars0 = ([hopf_phase_condition, pars])

for pars in parameter_array:
    sol = np.array(fsolve(shooting(hopf_bif), u0, args=initial_pars0))
    sol_list.append(sol)
    u0 = sol