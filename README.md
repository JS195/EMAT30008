# ODE and PDE solver library
A library for solving Ordinary Differential Equations (ODEs) and Partial Differential Equations (PDEs) using various numerical methods.

## Setup

1. Clone the project repository from GitHub.
2. Install the required dependencies.

## Dependencies
The following libraries and dependencies are required to run this project:

- Numpy
- Matplotlib
- Scipy
- Math
- Time

**Note:** The specific version numbers may vary depending on your environment and requirements.

## Usage
1. Choose the appropriate solver function based on your problem type.
2. Specify the equations, initial/boundary conditions, and integration parameters.
3. Obtain and analyse the numerical solutions. Customise functions for specific problem domains.


# Further Description of the Files

## Example Functions
Various useful functions to test the code with. Your own functions can also be specified. 

## ODE Solver
The ode_solve function in the ODE and PDE solver library solves ODEs using numerical methods. The user can specify the differential equation, initial conditions, and integration parameters to obtain a numerical solution.

## Numerical Shooting
The numerical_shooting function in the ODE and PDE solver library solves BVPs by iteratively adjusting initial conditions and solving the resulting ODEs. This method involves shooting from the boundary conditions and adjusting the initial guess until the solution matches the other boundary condition.

## Numerical Continuation
The numerical_continuation function in the ODE and PDE solver library solves BVPs by systematically varying a parameter in the problem. This method involves starting with a simple problem and incrementally increasing the complexity until the desired solution is obtained.

## Boundary Value Problem Solver
This code solves one-dimensional boundary value problems (BVPs) using finite difference methods for various boundary conditions (Dirichlet, Neumann, and Robin) and source functions. It provides multiple helper functions to construct and solve the linear system of equations representing the BVP, as well as to visualise the numerical and exact solutions.

## PDE Solver
The pde_solve function in the ODE and PDE solver library solves PDEs using numerical methods. The user can specify the PDE, initial and boundary conditions, and integration parameters to obtain a numerical solution.