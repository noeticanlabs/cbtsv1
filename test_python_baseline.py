#!/usr/bin/env python3
"""
Python Baseline Minkowski Stability Test
Run the stability test using Python GR stepper directly.
"""

import numpy as np
import time
from gr_solver.gr_stepper import GRStepper
from gr_solver.gr_core_fields import GRFields, SYM6_IDX
from gr_solver.gr_constraints import GRConstraints

def test_python_baseline():
    # Initialize Minkowski fields
    Nx, Ny, Nz = 10, 10, 10
    dx, dy, dz = 0.1, 0.1, 0.1
    fields = GRFields(Nx, Ny, Nz, dx, dy, dz)
    fields.init_minkowski()

    stepper = GRStepper(fields)
    constraints = GRConstraints(stepper.fields)

    # Add perturbation ε=1e-6
    epsilon = 1e-6
    k = 10.0
    x = np.arange(Nx) * dx - (Nx * dx) / 2
    y = np.arange(Ny) * dx - (Ny * dx) / 2
    z = np.arange(Nz) * dx - (Nz * dx) / 2
    X, Y, Z_mesh = np.meshgrid(x, y, z, indexing='ij')
    fields.gamma_sym6[..., SYM6_IDX["xx"]] += epsilon * np.sin(k * X) * np.sin(k * Y) * Z_mesh
    fields.gamma_sym6 += epsilon * np.random.randn(*fields.gamma_sym6.shape)

    # Initial constraints
    constraints.compute()
    initial_eps_H = np.max(np.abs(constraints.eps_H))
    initial_eps_M = np.max(np.abs(constraints.eps_M))
    print(f"Initial eps_H_max: {initial_eps_H:.2e}, eps_M_max: {initial_eps_M:.2e}")

    # Run 100 steps (T=1.0, dt=0.01)
    dt = 0.01
    T_max = 1.0
    t = 0.0
    step = 0
    max_eps_H = initial_eps_H
    max_eps_M = initial_eps_M

    start_time = time.time()
    while t < T_max:
        rhs = stepper.step(dt, lambda_val=0.0, sources_enabled=False)

        for key in rhs:
            if key in fields:
                fields[key] += dt * rhs[key]

        step += 1
        t += dt

        # Compute constraints
        constraints.compute()
        eps_H_max = np.max(np.abs(constraints.eps_H))
        eps_M_max = np.max(np.abs(constraints.eps_M))

        max_eps_H = max(max_eps_H, eps_H_max)
        max_eps_M = max(max_eps_M, eps_M_max)

        if step % 10 == 0:
            print(f"Step {step}: eps_H_max {eps_H_max:.2e}, eps_M_max {eps_M_max:.2e}")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

    # Final
    eps_H_final = np.max(np.abs(constraints.eps_H))
    eps_M_final = np.max(np.abs(constraints.eps_M))

    # Growth
    max_eps_H_growth = max_eps_H - initial_eps_H
    max_eps_M_growth = max_eps_M - initial_eps_M
    print(f"Max eps_H during run: {max_eps_H:.2e}, growth: {max_eps_H_growth:.2e}")
    print(f"Max eps_M during run: {max_eps_M:.2e}, growth: {max_eps_M_growth:.2e}")
    print(f"Final eps_H_max: {eps_H_final:.2e}, eps_M_max: {eps_M_final:.2e}")

    assert eps_H_final < 1e-2, f"Hamiltonian constraint not stable: {eps_H_final}"
    assert eps_M_final < 1e-2, f"Momentum constraint not stable: {eps_M_final}"
    print("Python baseline Minkowski stability test passed.")
    return execution_time, initial_eps_H, initial_eps_M, eps_H_final, eps_M_final, max_eps_H_growth, max_eps_M_growth

if __name__ == "__main__":
    metrics = test_python_baseline()
    print("Python Baseline Metrics:", metrics)