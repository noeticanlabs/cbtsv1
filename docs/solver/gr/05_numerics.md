# Numerical Methods

## Time Integration

The solver supports multiple time integration schemes:

- **Runge-Kutta 4 (RK4)** - 4th order accurate
- **Runge-Kutta 3 (RK3)** - 3rd order accurate  
- **Forward Euler** - 1st order (for testing)
- **Symplectic integrators** - for long-term stability

See [`symplectic.py`](../../src/cbtsv1/numerics/symplectic.py) for symplectic implementations.

## Spatial Discretization

- **Finite Differencing**: Various stencil orders (2nd, 4th, 6th order)
- **Spectral Methods**: See [`spectral/`](../../src/cbtsv1/numerics/spectral/)
- **Boundary Conditions**: Robin, Dirichlet, Neumann, Sommerfeld

## Constraint Solving

- **Elliptic Solver**: Multigrid methods for maximal slicing
- See [`multigrid.py`](../../src/cbtsv1/numerics/multigrid.py)

## dissipation

Kreiss-Oliger dissipation is applied to high-frequency modes to stabilize the evolution.

## Parallelization

The solver supports:
- Multi-threaded execution via phaseloom orchestrator
- Domain decomposition
- OpenMP/threading
