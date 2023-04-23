import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
# from ExampleFunctions import hopf_bif, hopf_bif_pc, cubic  # Commented out since the functions are defined below
# from NumericalShooting import shooting, find_shoot_orbit  # Commented out since the shooting function is defined below


def shooting(f, phase_cond):
    def G(U0T, param):
        U0, T = U0T[:-1], U0T[-1]
        x0 = np.hstack((U0, T, param))
        F = fsolve(phase_cond, x0, param)
        return np.hstack((F, 0))

    return G


def hopf_bif(t, X, pars):
    (u1, u2) = X
    du1dt = (pars * u1) - u2 - (u1 * (u1**2 + u2**2))
    du2dt = u1 + (pars * u2) - (u2 * (u1**2 + u2 ** 2))
    return np.array([du1dt, du2dt])


def hopf_bif_pc(x0, pars):
    return hopf_bif(0, x0, pars)[0]  # Changed 'x0' to '0' as time parameter 't'


def pseudo_arclength_continuation_shooting(f, phase_cond, U0T, initial_param, final_param, step_size, max_iter=100, tol=1e-8):
    param = initial_param
    U0T = np.hstack((U0T, 0))
    branch = []

    while param < final_param:
        for _ in range(max_iter):
            G = shooting(f, phase_cond)
            delta_U0T = fsolve(G, U0T, param)
            U0T += delta_U0T
            param += step_size

            if np.linalg.norm(delta_U0T) < tol:
                break

        branch.append((U0T, param))
        param += step_size

    return branch


# Set up parameters
initial_param = 0
final_param = 10
step_size = 0.1
U0T = np.zeros(2)  # Change this to the appropriate initial condition and period

# Solve
branch = pseudo_arclength_continuation_shooting(hopf_bif, hopf_bif_pc, U0T, initial_param, final_param, step_size)
