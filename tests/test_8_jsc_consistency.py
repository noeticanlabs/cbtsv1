"""
Test 8 JSC Consistency
"""

import numpy as np
import logging

class Test8Jsc:
    def __init__(self, gr_solver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        # Initialize solver with Minkowski
        self.gr_solver.init_minkowski()
        self.gr_solver.geometry.compute_christoffels()
        self.gr_solver.geometry.compute_ricci()
        self.gr_solver.geometry.compute_scalar_curvature()

        # Store original fields
        original_gamma = self.gr_solver.fields.gamma_sym6.copy()
        original_K = self.gr_solver.fields.K_sym6.copy()

        # Pick random perturbation δu (random arrays for gamma_sym6 and K_sym6)
        np.random.seed(42)
        delta_gamma = np.random.randn(*self.gr_solver.fields.gamma_sym6.shape) * 1e-10
        delta_K = np.random.randn(*self.gr_solver.fields.K_sym6.shape) * 1e-10

        # Compute F(u) = stepper.compute_rhs()
        self.gr_solver.stepper.rhs_computer.compute_rhs(0.0, slow_update=False)
        rhs_computer = self.gr_solver.stepper.rhs_computer
        F_u_gamma = rhs_computer.rhs_gamma_sym6.copy()
        F_u_K = rhs_computer.rhs_K_sym6.copy()
        F_u_phi = rhs_computer.rhs_phi.copy()

        ratios = []
        epsilons = [1e-6, 1e-7, 1e-8]
        for eps in epsilons:
            # Perturb fields: u + ε δu
            self.gr_solver.fields.gamma_sym6 = original_gamma + eps * delta_gamma
            self.gr_solver.fields.K_sym6 = original_K + eps * delta_K

            # Recompute geometry since gamma changed
            self.gr_solver.geometry.compute_christoffels()
            self.gr_solver.geometry.compute_ricci()
            self.gr_solver.geometry.compute_scalar_curvature()

            # Compute F(u + ε δu)
            self.gr_solver.stepper.rhs_computer.compute_rhs(0.0, slow_update=False)
            F_eps_gamma = rhs_computer.rhs_gamma_sym6.copy()
            F_eps_K = rhs_computer.rhs_K_sym6.copy()
            F_eps_phi = rhs_computer.rhs_phi.copy()

            # Compute ||F(u + ε δu) - F(u)|| / ε
            diff_gamma = F_eps_gamma - F_u_gamma
            diff_K = F_eps_K - F_u_K
            diff_phi = F_eps_phi - F_u_phi
            norm_diff = np.sqrt(np.sum(diff_gamma**2) + np.sum(diff_K**2) + np.sum(diff_phi**2))
            ratio = norm_diff / eps
            ratios.append(ratio)

        # Restore original fields
        self.gr_solver.fields.gamma_sym6 = original_gamma
        self.gr_solver.fields.K_sym6 = original_K
        self.gr_solver.geometry.compute_christoffels()
        self.gr_solver.geometry.compute_ricci()
        self.gr_solver.geometry.compute_scalar_curvature()

        # Check if ratios converge (approximately constant or decreasing for small ε)
        # Pass if the ratio is consistent for small ε (e.g., not diverging)
        passed = ratios[2] >= 0.5 * ratios[1] and ratios[2] <= 2 * ratios[1] and ratios[1] >= 0.5 * ratios[0] and ratios[1] <= 2 * ratios[0]

        diagnosis = f"Ratios for ε=[1e-6,1e-7,1e-8]: {ratios[0]:.2e}, {ratios[1]:.2e}, {ratios[2]:.2e} - {'consistent' if passed else 'inconsistent'}"

        return {'passed': passed, 'metrics': {'ratios': ratios}, 'diagnosis': diagnosis}