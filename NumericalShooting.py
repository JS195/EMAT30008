import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import fsolve
from ExampleFunctions import predator_prey, hopf, three_dim_hopf, standard_pc
from ODEsolver import solve_odes, plot_different_parameters, plotter
from scipy.spatial.distance import sqeuclidean

def plot_phase_portrait(func=predator_prey,  x0=[1, 1], t0=0, t1=200, dt_max=0.01, solver='rk4', **kwargs):
    """
    Plots the phase portrait of an ODE or ODE system.
    
    :param func: The function defining the ODE or ODE system.
    :param x0: The initial condition for the ODE system.
    :param t0: The initial time for the ODE system.
    :param t1: The final time for the ODE system.
    :param dt_max: The maximum time step size allowed by the solver.
    :param solver: The ODE solver to use. Should be one of 'rk4' (default) or 'euler'.
    :param **kwargs: Optional. Additional keyword arguments to pass to the ODE solver.

    :returns: None, but produces a plot of the phase portrait.
    """
    sol, t = solve_odes(func, x0=x0, t0=t0, t1=t1, dt_max=dt_max, solver=solver, **kwargs)
    fig, ax = plt.subplots()
    plotter(sol[:, 0], sol[:, 1], '-', 'x', 'y', 'Predator-Prey Phase Portrait', ax)
    plt.show()

def iso_orbit(f, x0, t0, t1, dt_max, atol=1e-4, **kwargs):
    """
    Finds the limit cycle initial conditions and time period for a system of ODEs using the Runge-Kutta 
    4th order solver.

    :param f: Function defining a system of ODEs.
    :param x0: Starting value of the dependent variable(s).
    :param t0: Starting time value.
    :param t1: Final time value.
    :param dt_max: Maximum step size.
    :param **kwargs: Optional. Any additional input keyword arguments.
    
    :returns: A list containing the initial conditions, and the time period of the limit cycle, if 
    one exists. If no limit cycle is found, returns None.
    """    
    # Some ValueErrors
    if dt_max <= 0:
        raise ValueError("dt_max must be greater than 0")
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0")
    if not isinstance(x0, (list, tuple, np.ndarray)):
        raise ValueError("x0 must be a list, tuple, or numpy array")

    # Solve ODE using rk4 method and extract the solution
    sol, t = solve_odes(f, x0, t0, t1, dt_max, 'rk4', **kwargs)
    # Find the indices of the peaks in the first coordinate
    peak_indices = np.where((sol[1:-1, 0] > sol[:-2, 0]) & (sol[1:-1, 0] > sol[2:, 0]))[0] + 1
    # Check for a limit cycle
    for i in range(1, len(peak_indices)):
        # Calculate the Euclidean distance between consecutive peaks
        distance = sqeuclidean(sol[peak_indices[i]], sol[peak_indices[i - 1]])
        if np.isclose(distance, 0, atol=atol):
            # Calculate time period of limit cycle
            period = t[peak_indices[i]] - t[peak_indices[i - 1]]
            # Store the n-dimensional state and time period in a list
            orbit_info = list(sol[peak_indices[i]]) + [period]
            return orbit_info

    # If no limit cycle is found, raise a warning and return None
    warnings.warn("No limit cycle found within the given absolute tolerance", UserWarning)
    return None

def shooting(f, phase_cond):
    """
    Returns a function G for solving a boundary value problem using the shooting method.

    :param f: System of ODEs function.
    :param phase_cond: Boundary conditions function.
    :returns: Function G calculating differences between actual and guessed boundary conditions.
    """
    if not callable(f):
        raise ValueError("f must be a callable function")
    if not callable(phase_cond):
        raise ValueError("phase_cond must be a callable function")
    
    def G(u0, T, pars):
        """
        Calculates differences between actual and guessed boundary conditions for a given problem.

        :param u0: Initial guess for the solution.
        :param T: Final time value.
        :param pars: Dictionary of parameter values.
        :returns: Numpy array of differences between actual and guessed boundary conditions.
        """
        # Ignore runTime warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # Solve ODE system using rk4 method
        sol, t = solve_odes(f, x0=u0, t0=0, t1=T, dt_max=0.01, solver='rk4', pars=pars)
        final_sol = sol[-1, :]
        # Calculate differences between actual and estimates
        return np.append(u0 - final_sol, phase_cond(f, u0, pars=pars))

    return G

def find_shoot_orbit(f, u0T, pars, phase_cond=standard_pc):
    """
    Finds the periodic orbit of a dynamical system using the shooting method.

    :param f: Function defining the dynamical system.
    :param phase_cond: Function defining the phase condition of the periodic orbit.
    :param u0T: Initial guess for the period and the initial condition of the orbit.
    :param pars: Parameters for the dynamical system.

    :returns: The initial conditions and the time period or the orbit as an array.
    """
    # Get G function for solving BVP using shooting
    G = shooting(f, phase_cond)
    #Use fsolve to find the root of G
    orbit = fsolve(lambda u0T: G(u0T[:-1], u0T[-1], pars), u0T)
    return orbit

def code_testing(func, x0, pars, u0T, atol=1e-2):
    """
    Tests the accuracy of the shooting method for finding periodic orbits by comparing the 
    results with the iso_orbit.

    :param func: The function defining the dynamical system.
    :param pars: Parameters for the dynamical system.
    :param atol: The desired tolerance, defaults to 1e-2.

    :returns: None, but prints "Test Passed" if the difference between the computed orbits is
              below the tolerance or "Test Failed" if the difference is above the tolerance.
    """
    orbit = iso_orbit(func, x0, 0, 200, 0.01, pars=pars)
    # Using the true values from before to provide an initial guess
    u0T = [orbit]
    shooting_orbit = find_shoot_orbit(func, u0T, pars)

    # Ensure both arrays have the same length
    if len(orbit) != len(shooting_orbit):
        print('Error: length mismatch between orbit and shooting_orbit.')

    test_passed = True  # Initialize the flag variable as True
    for i in range(len(shooting_orbit)):
        result = abs(orbit[i]-shooting_orbit[i])
        if result > atol:
            test_passed = False  # Update the flag variable to False if a comparison fails
    if test_passed:
        print("Test Passed")
    else:
        print("Test Failed")

def main():
    params = [[1.0, 0.1, 0.1], [1.0, 0.25, 0.1], [1.0, 0.4, 0.1]]
    plot_different_parameters(predator_prey, x0=[1,1], t0=0, t1=200, dt_max=0.01, params=params)
    
    #Plotting phase portrait
    pars = [1.0, 0.2, 0.1]
    plot_phase_portrait(pars=pars)

    # Isolate periodic orbit
    orbit = iso_orbit(predator_prey, x0=[1,1], t0=0, t1=500, dt_max=0.01, pars=[1.0,0.2,0.1])
    print('The true values of the predator prey orbit:', orbit)

    #Predator Prey shooting/ root finding
    # Using the true values from before to provide an initial guess
    u0T = orbit
    shooting_orbit = find_shoot_orbit(predator_prey, u0T, pars)
    print('The shooting values of the predator prey orbit: ', shooting_orbit)

    #Testing the shooting code using the supercritical Hopf bifurcation
    pars = [0.3, -1]
    orbit = iso_orbit(hopf, [1,1], 0, 200, 0.01, pars=pars)
    print('The true values of the hopf orbit:', orbit)

    # Using the true values from before to provide an initial guess
    u0T = [0.6, 0.001, 6]
    shooting_orbit = find_shoot_orbit(hopf, u0T, pars)
    print('The shooting values of the hopf orbit: ', shooting_orbit)

    #Testing the shooting code using the three dimensional hopf system
    pars = [0.3, -1]
    orbit = iso_orbit(three_dim_hopf, [1,1,1], 0, 200, 0.01, pars=pars)
    print('The true values of the three dim hopf orbit:', orbit)

    # Using the true values from before to provide an initial guess
    u0T = [0.6, 0.001, 0.001, 6]
    shooting_orbit = find_shoot_orbit(three_dim_hopf, u0T, pars)
    print('The shooting values of the three dim hopf orbit: ', shooting_orbit)


if __name__ == "__main__":
    main()