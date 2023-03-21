import numpy as np
from matplotlib import pyplot
import phaseportraits as pp

def morris_lecar(t, u, I_ext):
    # Taken from Table 1 of doi:10.1016/j.neucom.2005.03.006
    C_M = 20
    g_K = 8
    g_L = 2
    V_Ca = 120
    V_K = -80
    V_L = -60
    V_1 = -1.2
    V_2 = 18
    # Taken from Table 2 (class I) of doi:10.1016/j.neucom.2005.03.006
    g_Ca = 4.0
    phi = 1/15
    V_3 = 12
    V_4 = 17.4
    # References from doi:10.1016/j.neucom.2005.03.006
    (V, N) = u
    M_inf = 0.5*(1 + np.tanh((V - V_1)/V_2)) # (2)
    N_inf = 0.5*(1 + np.tanh((V - V_3)/V_4)) # (3)
    tau_N = 1/(phi*np.cosh((V - V_3)/(2*V_4))) # (4)
    # (1)
    dVdt = (-g_L*(V - V_L) - g_Ca*M_inf*(V - V_Ca) - g_K*N*(V - V_K) + I_ext)/C_M
    dNdt = (N_inf - N)/tau_N
    return np.array((dVdt, dNdt))

ml30 = lambda t, u: morris_lecar(t, u, 30)

# Some orbits
(t, u) = pp.orbit(ml30, (0, 0), 100)
pyplot.plot(u[0, :], u[1, :], "b-")
(t, u) = pp.orbit(ml30, (-20, 0), 100)
pyplot.plot(u[0, :], u[1, :], "b-")
(t, u) = pp.orbit(ml30, (20, 0), 100)
pyplot.plot(u[0, :], u[1, :], "b-")
(t, u) = pp.orbit(ml30, (0, 0.5), 100)
pyplot.plot(u[0, :], u[1, :], "b-")
(t, u) = pp.orbit(ml30, (0, 1.0), 100)
pyplot.plot(u[0, :], u[1, :], "b-")

# V nullcline
(Vval, Nval) = pp.nullcline(ml30, (-60, 20), index=0)
pyplot.plot(Vval, Nval, "g-")

# N nullcline
(Vval, Nval) = pp.nullcline(ml30, (-60, 20), index=1)
pyplot.plot(Vval, Nval, "r-")

# Equilibria
u = pp.equilibrium(ml30, (-40, 0))
pyplot.plot(u[0], u[1], "k.")
u = pp.equilibrium(ml30, (-20, 0))
pyplot.plot(u[0], u[1], "k.")
u = pp.equilibrium(ml30, (5, 0.3))
pyplot.plot(u[0], u[1], "k.")

pyplot.show()
