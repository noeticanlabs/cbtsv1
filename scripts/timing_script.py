import time
from src.core.gr_solver import GRSolver

def W0():
    """Small grid GR solver workload."""
    Nx, Ny, Nz = 8, 8, 8
    T_max = 1e-9
    solver = GRSolver(Nx, Ny, Nz, dx=1.0, dy=1.0, dz=1.0, c=1.0, Lambda=0.0)
    solver.init_minkowski()
    start = time.perf_counter()
    solver.run(T_max=T_max)
    end = time.perf_counter()
    elapsed = end - start
    print(f"W0 execution time: {elapsed:.4f} seconds")

def W1():
    """Medium grid GR solver workload."""
    Nx, Ny, Nz = 12, 12, 12
    T_max = 1e-8
    solver = GRSolver(Nx, Ny, Nz, dx=1.0, dy=1.0, dz=1.0, c=1.0, Lambda=0.0)
    solver.init_minkowski()
    start = time.perf_counter()
    solver.run(T_max=T_max)
    end = time.perf_counter()
    elapsed = end - start
    print(f"W1 execution time: {elapsed:.4f} seconds")

def W2():
    """Large grid GR solver workload."""
    Nx, Ny, Nz = 16, 16, 16
    T_max = 1e-8
    solver = GRSolver(Nx, Ny, Nz, dx=1.0, dy=1.0, dz=1.0, c=1.0, Lambda=0.0)
    solver.init_minkowski()
    start = time.perf_counter()
    solver.run(T_max=T_max)
    end = time.perf_counter()
    elapsed = end - start
    print(f"W2 execution time: {elapsed:.4f} seconds")

if __name__ == '__main__':
    print("Running timing script for GR solver workloads...")
    W0()
    W1()
    W2()
    print("Timing complete.")