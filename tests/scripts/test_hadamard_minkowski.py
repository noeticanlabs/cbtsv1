#!/usr/bin/env python3
"""
Hadamard Pipeline Minkowski Stability Test
Run the stability test using Hadamard VM for RHS computation.
"""

import numpy as np
import json
import time
from gr_solver.gr_core_fields import det_sym6, SYM6_IDX
from nsc_runtime_min import make_rhs_callable
from gr_constraints_nsc import compute_hamiltonian_compiled, compute_momentum_compiled
from gr_geometry_nsc import compute_christoffels_compiled, compute_ricci_compiled, compute_ricci_scalar_compiled

def test_hadamard_minkowski_stability():
    # Load Minkowski NSC IR for Hadamard
    rhs_func = make_rhs_callable('minkowski_rhs.nscir.json')

    # Initialize Minkowski fields (same as gr_core_fields.init_minkowski)
    Nx, Ny, Nz = 10, 10, 10
    dx, dy, dz = 0.1, 0.1, 0.1
    gamma_sym6 = np.zeros((Nx, Ny, Nz, 6))
    gamma_sym6[..., SYM6_IDX["xx"]] = 1.0
    gamma_sym6[..., SYM6_IDX["yy"]] = 1.0
    gamma_sym6[..., SYM6_IDX["zz"]] = 1.0
    K_sym6 = np.zeros((Nx, Ny, Nz, 6))
    alpha = np.ones((Nx, Ny, Nz))
    beta = np.zeros((Nx, Ny, Nz, 3))
    phi = np.zeros((Nx, Ny, Nz))
    gamma_tilde_sym6 = gamma_sym6.copy()
    A_sym6 = np.zeros((Nx, Ny, Nz, 6))
    Gamma_tilde = np.zeros((Nx, Ny, Nz, 3))
    Z = np.zeros((Nx, Ny, Nz))
    Z_i = np.zeros((Nx, Ny, Nz, 3))

    # Check initial constraints
    christoffels, _ = compute_christoffels_compiled(gamma_sym6, dx, dy, dz)
    ricci_full = compute_ricci_compiled(gamma_sym6, christoffels, dx, dy, dz)
    R_scalar = compute_ricci_scalar_compiled(ricci_full, gamma_sym6)
    eps_H = compute_hamiltonian_compiled(R_scalar, gamma_sym6, K_sym6, 0.0)
    eps_M = compute_momentum_compiled(gamma_sym6, K_sym6, christoffels, dx, dy, dz)
    initial_eps_H_max = np.max(np.abs(eps_H))
    initial_eps_M_max = np.max(np.abs(eps_M))
    print(f"Initial eps_H_max {initial_eps_H_max:.2e}, eps_M_max {initial_eps_M_max:.2e}")

    # Add small perturbations
    epsilon = 1e-6
    k = 10.0
    x = np.arange(Nx) * dx - (Nx * dx) / 2
    y = np.arange(Ny) * dy - (Ny * dy) / 2
    z = np.arange(Nz) * dz - (Nz * dz) / 2
    X, Y, Z_mesh = np.meshgrid(x, y, z, indexing='ij')
    gamma_sym6[..., SYM6_IDX["xx"]] += epsilon * np.sin(k * X) * np.sin(k * Y) * np.sin(k * Z_mesh)
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

    # Run 100 steps (T=1.0, dt=0.01)
    T_max = 1.0
    dt = 0.01
    t = 0.0
    step = 0
    max_eps_H = initial_eps_H_max
    max_eps_M = initial_eps_M_max

    start_time = time.time()
    while t < T_max:
        # Compute RHS using Hadamard VM
        rhs = rhs_func(fields, lambda_val=0.0, sources_enabled=False)

        # Euler step
        for key in rhs:
            if key in fields:
                fields[key] += dt * rhs[key]

        step += 1
        t += dt

        # Compute constraints
        christoffels, _ = compute_christoffels_compiled(fields['gamma_sym6'], dx, dy, dz)
        ricci_full = compute_ricci_compiled(fields['gamma_sym6'], christoffels, dx, dy, dz)
        R_scalar = compute_ricci_scalar_compiled(ricci_full, fields['gamma_sym6'])
        eps_H = compute_hamiltonian_compiled(R_scalar, fields['gamma_sym6'], fields['K_sym6'], 0.0)
        eps_M = compute_momentum_compiled(fields['gamma_sym6'], fields['K_sym6'], christoffels, dx, dy, dz)

        eps_H_max_step = np.max(np.abs(eps_H))
        eps_M_max_step = np.max(np.abs(eps_M))

        max_eps_H = max(max_eps_H, eps_H_max_step)
        max_eps_M = max(max_eps_M, eps_M_max_step)

        if step % 10 == 0:
            print(f"Step {step}: eps_H_max {eps_H_max_step:.2e}, eps_M_max {eps_M_max_step:.2e}")

        # Check stability (allow some growth)
        if eps_H_max_step > 2e-3 or eps_M_max_step > 2e-3:
            print(f"Constraints violated at step {step}")
            break

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

    # Final check
    eps_H_final = np.max(np.abs(eps_H))
    eps_M_final = np.max(np.abs(eps_M))

    # Growth
    max_eps_H_growth = max_eps_H - initial_eps_H_max
    max_eps_M_growth = max_eps_M - initial_eps_M_max
    print(f"Max eps_H during run: {max_eps_H:.2e}, growth: {max_eps_H_growth:.2e}")
    print(f"Max eps_M during run: {max_eps_M:.2e}, growth: {max_eps_M_growth:.2e}")

    print(f"Final eps_H_max: {eps_H_final:.2e}, eps_M_max: {eps_M_final:.2e}")
    assert eps_H_final < 2e-3, f"Hamiltonian constraint not stable: {eps_H_final}"
    assert eps_M_final < 2e-3, f"Momentum constraint not stable: {eps_M_final}"
    print("Hadamard Minkowski stability test passed.")
    return execution_time, initial_eps_H_max, initial_eps_M_max, eps_H_final, eps_M_final, max_eps_H_growth, max_eps_M_growth

if __name__ == "__main__":
    metrics = test_hadamard_minkowski_stability()
    print("Hadamard Metrics:", metrics)