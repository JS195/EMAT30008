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

def pseudo_arclength_continuation(f, u0, min_par, max_par, no_steps, step_size, phase_cond=None, discretisation='shooting'):

    def arclength_residual(u_par, u0_par, f, step_size):
        u = u_par[:-1]
        par = u_par[-1]
        res = np.hstack((f(u, 0, par), np.linalg.norm(u_par - u0_par) - step_size))
        return res

    sol_list = [u0]
    par_list = [min_par]

    for _ in range(10):
        if par_list[-1] >= max_par:
            break
        if phase_cond is not None and discretisation == 'shooting':
            raise NotImplementedError("Shooting method is not implemented for pseudo arclength continuation")
        else:
            u0_par = np.hstack((sol_list[-1], par_list[-1]))
            u_par = fsolve(arclength_residual, u0_par, args=(u0_par, f, step_size))
            sol_list.append(u_par[:-1])
            par_list.append(u_par[-1])

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
