import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gr_solver.gr_solver import GRSolver
from gr_solver.gr_core_fields import SYM6_IDX

def test_momentum_constraint_sanity():
    """Sanity test for momentum constraint: perturb K_ij and verify eps_M scales with perturbation."""

    # Initialize solver
    Nx, Ny, Nz = 16, 16, 16
    dx = 1.0
    solver = GRSolver(Nx, Ny, Nz, dx, dx, dx)
    solver.init_minkowski()

    # Compute initial constraints
    solver.constraints.compute_hamiltonian()
    solver.constraints.compute_momentum()
    solver.constraints.compute_residuals()

    eps_M_initial = solver.constraints.eps_M
    print(f"Initial eps_M: {eps_M_initial}")

    # Test different perturbation sizes
    perturbation_sizes = [1e-6, 1e-5, 1e-4, 1e-3]
    eps_M_values = []

    for delta in perturbation_sizes:
        # Reset to Minkowski
        solver.init_minkowski()

        # Inject tiny random perturbation into K_ij
        np.random.seed(42)  # For reproducibility
        perturbation = np.random.normal(0, delta, solver.fields.K_sym6.shape)
        solver.fields.K_sym6 += perturbation

        # Compute constraints
        solver.constraints.compute_hamiltonian()
        solver.constraints.compute_momentum()
        solver.constraints.compute_residuals()

        eps_M = solver.constraints.eps_M
        eps_M_values.append(eps_M)
        print(f"Perturbation size {delta}: eps_M = {eps_M}")

    # Verify eps_M is nonzero and scales with perturbation
    assert eps_M_initial < 1e-10, f"Initial eps_M too large: {eps_M_initial}"

    for i, eps_M in enumerate(eps_M_values):
        assert eps_M > 1e-12, f"eps_M not nonzero for delta={perturbation_sizes[i]}: {eps_M}"

    # Check scaling: eps_M should roughly scale linearly with delta
    ratios = [eps_M / delta for eps_M, delta in zip(eps_M_values, perturbation_sizes)]
    print(f"Scaling ratios (eps_M / delta): {ratios}")

    # The ratios should be roughly constant (within order of magnitude)
    mean_ratio = np.mean(ratios)
    for ratio in ratios:
        assert 0.1 * mean_ratio < ratio < 10 * mean_ratio, f"Scaling not linear: ratios {ratios}"

    print("Momentum constraint sanity test passed!")

if __name__ == "__main__":
    test_momentum_constraint_sanity()