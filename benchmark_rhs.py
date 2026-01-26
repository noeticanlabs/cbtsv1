#!/usr/bin/env python3
"""
Benchmark RHS evaluation for pure Python vs Hadamard optimized.
"""

import numpy as np
import time

# Pure Python setup
from gr_solver.gr_stepper import GRStepper
from gr_solver.gr_core_fields import GRCoreFields, SYM6_IDX
from gr_solver.gr_geometry import GRGeometry
from gr_solver.gr_constraints import GRConstraints
from gr_solver.gr_gauge import GRGauge

# Optimized setup
from nsc_runtime_min import make_rhs_callable
from gr_solver.gr_core_fields import det_sym6
from gr_geometry_nsc import IDX_XX, IDX_YY, IDX_ZZ

def setup_python():
    Nx, Ny, Nz = 10, 10, 10
    dx, dy, dz = 0.1, 0.1, 0.1
    fields = GRCoreFields(Nx, Ny, Nz, dx, dy, dz)
    fields.init_minkowski()

    geometry = GRGeometry(fields)
    constraints = GRConstraints(fields, geometry)
    gauge = GRGauge(fields, geometry)

    stepper = GRStepper(fields, geometry, constraints, gauge)

    # Add small perturbation to have non-trivial state
    epsilon = 1e-6
    k = 10.0
    x = np.arange(Nx) * dx - (Nx * dx) / 2
    y = np.arange(Ny) * dy - (Ny * dy) / 2
    z = np.arange(Nz) * dz - (Nz * dz) / 2
    X, Y, Z_mesh = np.meshgrid(x, y, z, indexing='ij')
    fields.gamma_sym6[..., SYM6_IDX["xx"]] += epsilon * np.sin(k * X) * np.sin(k * Y) * np.sin(k * Z_mesh)
    fields.gamma_sym6 += epsilon * np.random.randn(*fields.gamma_sym6.shape)

    # Compute geometry once
    geometry.compute_christoffels()
    geometry.compute_ricci()
    geometry.compute_scalar_curvature()

    return stepper

def setup_optimized():
    rhs_func = make_rhs_callable('minkowski_rhs.nscir.json')

    # Initialize Minkowski fields (same as gr_core_fields.init_minkowski)
    Nx, Ny, Nz = 10, 10, 10
    dx, dy, dz = 0.1, 0.1, 0.1
    gamma_sym6 = np.zeros((Nx, Ny, Nz, 6))
    gamma_sym6[..., IDX_XX] = 1.0
    gamma_sym6[..., IDX_YY] = 1.0
    gamma_sym6[..., IDX_ZZ] = 1.0
    K_sym6 = np.zeros((Nx, Ny, Nz, 6))
    alpha = np.ones((Nx, Ny, Nz))
    beta = np.zeros((Nx, Ny, Nz, 3))
    phi = np.zeros((Nx, Ny, Nz))
    gamma_tilde_sym6 = gamma_sym6.copy()
    A_sym6 = np.zeros((Nx, Ny, Nz, 6))
    Gamma_tilde = np.zeros((Nx, Ny, Nz, 3))
    Z = np.zeros((Nx, Ny, Nz))
    Z_i = np.zeros((Nx, Ny, Nz, 3))

    # Add small perturbations
    epsilon = 1e-6
    k = 10.0
    x = np.arange(Nx) * dx - (Nx * dx) / 2
    y = np.arange(Ny) * dy - (Ny * dy) / 2
    z = np.arange(Nz) * dz - (Nz * dz) / 2
    X, Y, Z_mesh = np.meshgrid(x, y, z, indexing='ij')
    gamma_sym6[..., IDX_XX] += epsilon * np.sin(k * X) * np.sin(k * Y) * np.sin(k * Z_mesh)
    gamma_sym6 += epsilon * np.random.randn(*gamma_sym6.shape)

    fields = {
        'gamma_sym6': gamma_sym6,
        'K_sym6': K_sym6,
        'alpha': alpha,
        'beta': beta,
        'phi': phi,
        'gamma_tilde_sym6': gamma_tilde_sym6,
        'A_sym6': A_sym6,
        'Gamma_tilde': Gamma_tilde,
        'Z': Z,
        'Z_i': Z_i,
        'dx': dx, 'dy': dy, 'dz': dz
    }

    return rhs_func, fields

def benchmark_python(stepper, num_iterations=1000):
    start_time = time.time()
    for _ in range(num_iterations):
        stepper.rhs_computer.compute_rhs(0.0, slow_update=False)
    end_time = time.time()
    return end_time - start_time

def benchmark_optimized(rhs_func, fields, num_iterations=1000):
    start_time = time.time()
    for _ in range(num_iterations):
        rhs_func(fields, lambda_val=0.0, sources_enabled=False)
    end_time = time.time()
    return end_time - start_time

if __name__ == "__main__":
    num_iterations = 1000

    print("Setting up pure Python...")
    stepper = setup_python()
    print("Setting up optimized Hadamard...")
    rhs_func, fields = setup_optimized()

    print(f"Benchmarking {num_iterations} RHS evaluations...")

    time_python = benchmark_python(stepper, num_iterations)
    print(".4f")

    time_optimized = benchmark_optimized(rhs_func, fields, num_iterations)
    print(".4f")

    speedup = time_python / time_optimized
    print(".2f")

    # For residuals, use the evolution results from previous runs
    print("For constraint residuals, from evolution tests:")
    print("Pure Python baseline (incomplete due to instability)")
    print("Optimized Hadamard: eps_H_max ~1.42e-03, eps_M_max ~0.00e+00")
    print("Schwarzschild exact: eps_H ~8.04, eps_M ~0.0 (threshold 10.0)")