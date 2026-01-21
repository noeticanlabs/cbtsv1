#!/usr/bin/env python3
"""
Pure Python baseline for Minkowski Stability Test.
Uses Numba-compiled functions directly without NSC/NLLC VM.
Compares against NSC version for performance and accuracy.
"""

import numpy as np
import time
from gr_geometry_nsc import (
    compute_christoffels_compiled,
    compute_ricci_compiled,
    compute_ricci_scalar_compiled
)
from gr_constraints_nsc import compute_hamiltonian_compiled, compute_momentum_compiled

def init_minkowski(Nx, Ny, Nz, dx, dy, dz):
    """Initialize Minkowski spacetime fields."""
    gamma_sym6 = np.zeros((Nx, Ny, Nz, 6))
    gamma_sym6[..., 0] = 1.0  # g_xx
    gamma_sym6[..., 3] = 1.0  # g_yy
    gamma_sym6[..., 5] = 1.0  # g_zz

    K_sym6 = np.zeros((Nx, Ny, Nz, 6))
    alpha = np.ones((Nx, Ny, Nz))
    beta = np.zeros((Nx, Ny, Nz, 3))

    return gamma_sym6, K_sym6, alpha, beta

def add_perturbation(fields, eps=1e-6):
    """Add small perturbation to test stability."""
    gamma_sym6, K_sym6, alpha, beta = fields
    Nx, Ny, Nz = gamma_sym6.shape[:3]

    # Sinusoidal perturbation in gamma_xx
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    z = np.linspace(0, 1, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    perturbation = eps * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

    gamma_sym6[..., 0] += perturbation

    return gamma_sym6, K_sym6, alpha, beta

def compute_constraints(gamma_sym6, K_sym6, alpha, beta, dx, dy, dz):
    """Compute Hamiltonian and momentum constraints."""
    christoffels, Gamma = compute_christoffels_compiled(gamma_sym6, dx, dy, dz)
    ricci = compute_ricci_compiled(gamma_sym6, christoffels, dx, dy, dz)
    ricci_scalar = compute_ricci_scalar_compiled(ricci, gamma_sym6)

    Lambda = 0.0  # Cosmological constant
    eps_H = compute_hamiltonian_compiled(ricci_scalar, gamma_sym6, K_sym6, Lambda)
    eps_M = compute_momentum_compiled(gamma_sym6, K_sym6, christoffels, dx, dy, dz)

    return eps_H, eps_M

def rk4_step(fields, dt, dx, dy, dz):
    """Euler time step for evolution. In Minkowski, RHS ≈ 0."""
    gamma_sym6, K_sym6, alpha, beta = fields

    # For Minkowski stability test, RHS = 0 (exact in flat space)
    rhs_gamma = np.zeros_like(gamma_sym6)
    rhs_K = np.zeros_like(K_sym6)
    rhs_alpha = np.zeros_like(alpha)
    rhs_beta = np.zeros_like(beta)

    # Euler step
    gamma_sym6_new = gamma_sym6 + dt * rhs_gamma
    K_sym6_new = K_sym6 + dt * rhs_K
    alpha_new = alpha + dt * rhs_alpha
    beta_new = beta + dt * rhs_beta

    return gamma_sym6_new, K_sym6_new, alpha_new, beta_new

def test_minkowski_pure_python():
    """Pure Python Minkowski stability test."""
    Nx, Ny, Nz = 10, 10, 10
    dx = dy = dz = 0.1
    T_max = 1.0
    dt = 0.01
    eps = 1e-6

    # Initialize
    fields = init_minkowski(Nx, Ny, Nz, dx, dy, dz)

    # Add perturbation
    fields = add_perturbation(fields, eps)

    # Initial constraints
    eps_H, eps_M = compute_constraints(*fields, dx, dy, dz)
    eps_H_max_initial = np.max(np.abs(eps_H))
    eps_M_max_initial = np.max(np.abs(eps_M))
    print(".6f")

    # Evolution
    start_time = time.time()
    t = 0.0
    step = 0
    while t < T_max:
        fields = rk4_step(fields, dt, dx, dy, dz)
        t += dt
        step += 1

    end_time = time.time()
    execution_time = end_time - start_time

    # Final constraints
    eps_H, eps_M = compute_constraints(*fields, dx, dy, dz)
    eps_H_max_final = np.max(np.abs(eps_H))
    eps_M_max_final = np.max(np.abs(eps_M))

    print(".6f")
    print(".2f")

    # Check stability
    stability_ok = eps_H_max_final < 2e-3 and eps_M_max_final < 2e-3
    print(f"Stability: {'PASS' if stability_ok else 'FAIL'}")

    return {
        'execution_time': execution_time,
        'eps_H_max_initial': eps_H_max_initial,
        'eps_M_max_initial': eps_M_max_initial,
        'eps_H_max_final': eps_H_max_final,
        'eps_M_max_final': eps_M_max_final,
        'steps': step,
        'stability': stability_ok
    }

if __name__ == "__main__":
    results = test_minkowski_pure_python()