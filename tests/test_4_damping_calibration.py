"""
Test 4 Damping Calibration
"""

import numpy as np
import logging
import copy

class Test4:
    def __init__(self, gr_solver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        # Initialize solver with Minkowski data
        self.gr_solver.init_minkowski()

        # Add small random perturbations to excite dynamics
        np.random.seed(42)
        pert_scale = 1e-12
        self.gr_solver.fields.gamma_sym6 += pert_scale * np.random.randn(*self.gr_solver.fields.gamma_sym6.shape)
        self.gr_solver.fields.K_sym6 += pert_scale * np.random.randn(*self.gr_solver.fields.K_sym6.shape)
        self.gr_solver.fields.alpha += pert_scale * np.random.randn(*self.gr_solver.fields.alpha.shape)
        self.gr_solver.fields.beta += pert_scale * np.random.randn(*self.gr_solver.fields.beta.shape)

        # Store initial fields for delta_state calculation
        initial_gamma = self.gr_solver.fields.gamma_sym6.copy()
        initial_K = self.gr_solver.fields.K_sym6.copy()
        initial_alpha = self.gr_solver.fields.alpha.copy()
        initial_beta = self.gr_solver.fields.beta.copy()

        # Lambda values
        lambda_vals = [0.0, 0.1, 0.2, 0.4]

        T = 0.1
        eps_H_curve = []
        delta_state_curve = []

        for lambda_val in lambda_vals:
            # Create a deep copy of the solver
            solver = copy.deepcopy(self.gr_solver)

            # Set damping parameter in stepper
            solver.stepper.lambda_val = lambda_val

            # Reset time and step
            solver.orchestrator.t = 0.0
            solver.orchestrator.step = 0

            # Run evolution for fixed T
            while solver.orchestrator.t < T:
                dt_max = T - solver.orchestrator.t
                dt, _, _ = solver.orchestrator.run_step(dt_max)

            # Compute final eps_H
            solver.constraints.compute_hamiltonian()
            eps_H = solver.constraints.eps_H
            eps_H_curve.append(eps_H)

            # Compute delta_state(t) = sum of L2 norms of field deltas
            delta_gamma = solver.fields.gamma_sym6 - initial_gamma
            delta_K = solver.fields.K_sym6 - initial_K
            delta_alpha = solver.fields.alpha - initial_alpha
            delta_beta = solver.fields.beta - initial_beta

            l2_gamma = np.sqrt(np.sum(delta_gamma**2))
            l2_K = np.sqrt(np.sum(delta_K**2))
            l2_alpha = np.sqrt(np.sum(delta_alpha**2))
            l2_beta = np.sqrt(np.sum(delta_beta**2))
            delta_state = l2_gamma + l2_K + l2_alpha + l2_beta
            delta_state_curve.append(delta_state)

        # Check monotonic decrease in final eps_H with increasing lambda
        monotonic = all(eps_H_curve[i] > eps_H_curve[i+1] for i in range(len(eps_H_curve)-1))

        # Check delta_state > 1e-12 (for the highest lambda, assuming it's not driven to 0)
        delta_ok = delta_state_curve[-1] > 1e-12

        passed = monotonic and delta_ok

        diagnosis = f"eps_H monotonic decrease: {monotonic}, delta_state > 1e-12: {delta_ok}"

        return {'passed': passed, 'metrics': {'eps_H_curve': eps_H_curve, 'delta_state_curve': delta_state_curve}, 'diagnosis': diagnosis}