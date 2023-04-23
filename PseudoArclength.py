import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from ExampleFunctions import hopf_bif, hopf_bif_pc, cubic

def three_dim_hopf(U, t, pars):
    u1, u2, u3 = U[0], U[1], U[2]
    beta, sigma = pars[0], pars[1]
    du1dt = beta * u1 -u2 + sigma*u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)
    du3dt = -u3
    return np.array([du1dt, du2dt, du3dt])

def three_dim_hopf_pc(x0, pars):
    return three_dim_hopf(x0, 0, pars)[0]

def hopf_bif(X, t, pars):
    (u1, u2) = X
    du1dt = (pars * u1) - u2 - (u1 * (u1**2 + u2**2))
    du2dt = u1 + (pars * u2) - (u2 * (u1**2 + u2 ** 2))
    return np.array([du1dt, du2dt])

def hopf_bif_pc(x0, pars):
    return hopf_bif(x0, 0, pars)[0]


def pseudo_arclength_continuation(f, u0, init_par, no_steps, step_size, t, phase_cond=None, discretisation='shooting'):
    def extended_system(u_par, u_prev, par_prev, f, phase_cond, step_size, t):
        u, par = u_par[:-1], u_par[-1]
        pars = (par,)
        F = np.append(f(u, t, *pars), np.linalg.norm(u_par - np.append(u_prev, par_prev)) - step_size)
        return F

    sol_list = []
    par_list = [init_par]
    sol_list.append(u0)
    u_prev, par_prev = np.array(u0), init_par
    for _ in range(no_steps):
        u_par_guess = np.append(u_prev, par_prev) + step_size * np.ones_like(np.append(u_prev, par_prev))

        u_par_sol = fsolve(extended_system, u_par_guess, args=(u_prev, par_prev, f, phase_cond, step_size, t))
        u_sol, par_sol = u_par_sol[:-1], u_par_sol[-1]
        sol_list.append(u_sol)
        par_list.append(par_sol)
        u_prev, par_prev = u_sol, par_sol

    return np.array(sol_list), np.array(par_list)

# Test the functions

# Example parameters
params = (1, 2)

# Example initial conditions
x0 = np.array([1, 2, 3])

# Test hopf_bif function
print("Hopf bif:")
print(hopf_bif(x0[:2], 0, params[:1]))

# Test pseudo_arclength_continuation function
u0 = np.array([1.0, 2.0])
init_par = 0.5
no_steps = 10
step_size = 0.1
t = 0
solutions, parameters = pseudo_arclength_continuation(hopf_bif, u0, init_par, no_steps, step_size, t)
print("Solutions:")
print(solutions)
print("Parameters:")
print(parameters)

import matplotlib.pyplot as plt

# Parameters for pseudo_arclength_continuation
u0 = np.array([0.5, 0.6])
init_par = 0.5
no_steps = 200
step_size = 0.01
t = 0

# Run pseudo_arclength_continuation for the Hopf bifurcation
solutions, parameters = pseudo_arclength_continuation(hopf_bif, u0, init_par, no_steps, step_size, t)

# Plot the bifurcation diagram
plt.figure()
plt.plot(parameters, solutions[:, 0], 'b.', label='u1')
plt.plot(parameters, solutions[:, 1], 'r.', label='u2')
plt.xlabel('Parameter')
plt.ylabel('Fixed Points')
plt.legend()
plt.title('Bifurcation Diagram for Hopf Equation')
plt.show()
