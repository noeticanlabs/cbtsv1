"""
Test 0: BS2 Evolution
"""

import numpy as np
import logging

class Test0Bs2:
    def __init__(self, gr_solver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        # Initialize solver with Minkowski data
        self.gr_solver.init_minkowski()
        # Record initial eps_H, eps_M, and inf norms of gamma_sym6, K_sym6, alpha, beta at t=0
        self.gr_solver.constraints.compute_hamiltonian()
        self.gr_solver.constraints.compute_momentum()
        self.gr_solver.constraints.compute_residuals()
        eps_H = self.gr_solver.constraints.eps_H
        eps_M = self.gr_solver.constraints.eps_M
        inf_gamma = np.max(np.abs(self.gr_solver.fields.gamma_sym6))
        inf_K = np.max(np.abs(self.gr_solver.fields.K_sym6))
        inf_alpha = np.max(np.abs(self.gr_solver.fields.alpha))
        inf_beta = np.max(np.abs(self.gr_solver.fields.beta))

        # Make copies of initial fields
        gamma0 = self.gr_solver.fields.gamma_sym6.copy()
        K0 = self.gr_solver.fields.K_sym6.copy()
        alpha0 = self.gr_solver.fields.alpha.copy()
        beta0 = self.gr_solver.fields.beta.copy()

        # Apply small perturbation to gamma_sym6 and K_sym6 to excite dynamics
        pert = 1e-12
        self.gr_solver.fields.gamma_sym6[0, 0, 0, 0] += pert
        self.gr_solver.fields.K_sym6[0, 0, 0, 0] += pert

        # Take one step using the solver's orchestrator.run_step()
        dt, dominant_thread, rail_violation = self.gr_solver.orchestrator.run_step()

        # Record deltas
        delta_gamma = self.gr_solver.fields.gamma_sym6 - gamma0
        delta_K = self.gr_solver.fields.K_sym6 - K0
        delta_alpha = self.gr_solver.fields.alpha - alpha0
        delta_beta = self.gr_solver.fields.beta - beta0

        # Compute omega_state_sum as sum of L2 norms of deltas
        omega_gamma = np.sqrt(np.sum(delta_gamma**2))
        omega_K = np.sqrt(np.sum(delta_K**2))
        omega_alpha = np.sqrt(np.sum(delta_alpha**2))
        omega_beta = np.sqrt(np.sum(delta_beta**2))
        omega_state_sum = omega_gamma + omega_K + omega_alpha + omega_beta

        # Pass if at least one delta > 1e-14
        max_delta = max(np.max(np.abs(delta_gamma)), np.max(np.abs(delta_K)), np.max(np.abs(delta_alpha)), np.max(np.abs(delta_beta)))
        passed = max_delta > 1e-14

        # Restore solver state
        self.gr_solver.fields.gamma_sym6 = gamma0
        self.gr_solver.fields.K_sym6 = K0
        self.gr_solver.fields.alpha = alpha0
        self.gr_solver.fields.beta = beta0
        self.gr_solver.orchestrator.t = 0.0
        self.gr_solver.orchestrator.step = 0

        # Prepare metrics and diagnosis
        metrics = {
            'eps_H': eps_H,
            'eps_M': eps_M,
            'inf_gamma': inf_gamma,
            'inf_K': inf_K,
            'inf_alpha': inf_alpha,
            'inf_beta': inf_beta,
            'omega_state_sum': omega_state_sum
        }
        diagnosis = "Evolution detected: solver responded to perturbation." if passed else "No evolution: fields unchanged within tolerance."

        return {'passed': passed, 'metrics': metrics, 'diagnosis': diagnosis}