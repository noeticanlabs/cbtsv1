import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import logging
import copy
from src.core.gr_solver import GRSolver

class TestGcatGr1:
    """
    Constraint damping validation: Evolve with increasing lambda values
    and ensure eps_H decreases monotonically.
    """
    def __init__(self, gr_solver: GRSolver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        # Initialize with Minkowski + small random perturbations to excite dynamics
        self.gr_solver.init_minkowski()
        np.random.seed(42)
        pert_scale = 1e-12
        self.gr_solver.fields.gamma_sym6 += pert_scale * np.random.randn(*self.gr_solver.fields.gamma_sym6.shape)
        self.gr_solver.fields.K_sym6 += pert_scale * np.random.randn(*self.gr_solver.fields.K_sym6.shape)

        # Store initial fields for delta_state calculation
        initial_state = {
            'gamma': self.gr_solver.fields.gamma_sym6.copy(),
            'K': self.gr_solver.fields.K_sym6.copy()
        }

        # Lambda values
        lambda_vals = [0.0, 0.1, 0.2, 0.4]
        T = 0.1
        eps_H_curve = []
        delta_state_curve = []

        for lambda_val in lambda_vals:
            # Create a new solver with the same initial perturbation
            solver = copy.deepcopy(self.gr_solver)

            # Set damping parameter
            solver.stepper.lambda_val = lambda_val

            # Reset time
            solver.orchestrator.t = 0.0
            solver.orchestrator.step = 0

            # Evolve
            while solver.orchestrator.t < T:
                dt_max = T - solver.orchestrator.t
                dt, _, _ = solver.orchestrator.run_step(dt_max)

            # Compute final eps_H
            solver.constraints.compute_hamiltonian()
            eps_H = solver.constraints.eps_H
            eps_H_curve.append(eps_H)

            # Compute delta_state(t) = sum of L2 norms of field deltas
            delta_gamma = np.linalg.norm(solver.fields.gamma_sym6 - initial_state['gamma'])
            delta_K = np.linalg.norm(solver.fields.K_sym6 - initial_state['K'])
            delta_state = delta_gamma + delta_K
            delta_state_curve.append(delta_state)

        # Check monotonic decrease: eps_H should decrease as lambda increases.
        # Using strict inequality as damping should have a noticeable effect.
        monotonic = all(eps_H_curve[i] > eps_H_curve[i+1] for i in range(len(eps_H_curve)-1))

        # Also check delta_state > 1e-12 for the highest lambda to ensure evolution happened.
        delta_state_ok = delta_state_curve[-1] > 1e-12 if delta_state_curve else False

        passed = monotonic and delta_state_ok

        metrics = {
            'lambda_vals': lambda_vals,
            'eps_H_curve': eps_H_curve,
            'delta_state_curve': delta_state_curve,
            'monotonic': monotonic,
            'delta_state_ok': delta_state_ok
        }
        if passed:
            diagnosis = f"Damping valid: eps_H decreases monotonically with lambda ({eps_H_curve}) and state evolves."
        else:
            diagnosis = f"Damping invalid: monotonic={monotonic} (eps_H: {eps_H_curve}), delta_state_ok={delta_state_ok} (final delta_state: {delta_state_curve[-1]:.2e})"

        return {'passed': passed, 'metrics': metrics, 'diagnosis': diagnosis}

if __name__ == "__main__":
    print("Executing test: tests/test_gcat_gr_1.py")
    import json
    logging.getLogger('gr_solver.sem').setLevel(logging.CRITICAL)
    solver = GRSolver(16, 16, 16, dx=1.0, dy=1.0, dz=1.0)
    print("Solver initialized. Running test...")
    result = TestGcatGr1(solver).run()
    print("Test finished. Result:")
    print(result)
    with open('receipts_gcat_gr_1.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("Receipt file written.")
