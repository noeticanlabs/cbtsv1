"""
Test 6 Shat Stability
"""

import numpy as np
import logging
import copy

class Test6Shat:
    def __init__(self, gr_solver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        # Define high bands: indices where at least one dimension is high (2)
        def is_high_band(idx):
            i = idx // 9
            j = (idx // 3) % 3
            k = idx % 3
            return i == 2 or j == 2 or k == 2

        high_band_indices = [idx for idx in range(27) if is_high_band(idx)]

        # Function to compute E_highk
        def compute_E_highk(omega):
            return np.sum(omega[high_band_indices])

        # Initialize solver with Minkowski
        self.gr_solver.init_minkowski()

        N = self.gr_solver.fields.Nx
        k = N // 4  # mid-frequency
        eps = 1e-6  # amplitude
        x = np.arange(N) * self.gr_solver.fields.dx
        y = np.arange(N) * self.gr_solver.fields.dy
        z = np.arange(N) * self.gr_solver.fields.dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        pert = eps * np.sin(k * X) * np.sin(k * Y) * np.sin(k * Z)
        self.gr_solver.fields.gamma_sym6[..., 0] += pert  # xx component

        # Reset time and step
        self.gr_solver.orchestrator.t = 0.0
        self.gr_solver.orchestrator.step = 0

        # Run with dealiasing ON (damping enabled)
        solver_on = copy.deepcopy(self.gr_solver)
        solver_on.stepper.damping_enabled = True
        solver_on.stepper.lambda_val = 0.1  # some damping
        E_highk_on = []
        for step in range(20):
            dt, _, _ = solver_on.orchestrator.run_step()
            omega = solver_on.orchestrator.render.compute_omega()
            E_highk_on.append(compute_E_highk(omega))

        # Run with dealiasing OFF (damping disabled)
        solver_off = copy.deepcopy(self.gr_solver)
        solver_off.stepper.damping_enabled = False
        E_highk_off = []
        for step in range(20):
            dt, _, _ = solver_off.orchestrator.run_step()
            omega = solver_off.orchestrator.render.compute_omega()
            E_highk_off.append(compute_E_highk(omega))

        # Pass if ON: E_highk slowly increases or decays; OFF: explodes
        # Check if ON is stable: final < initial * 2 or decreasing
        initial_on = E_highk_on[0] if E_highk_on else 0
        final_on = E_highk_on[-1] if E_highk_on else 0
        stable_on = final_on <= initial_on * 2 and (final_on <= initial_on or len(E_highk_on) < 2 or E_highk_on[-1] < E_highk_on[-2])

        # OFF explodes: final > initial * 10
        initial_off = E_highk_off[0] if E_highk_off else 0
        final_off = E_highk_off[-1] if E_highk_off else 0
        explodes_off = final_off > initial_off * 10

        passed = stable_on and explodes_off

        diagnosis = f"ON: stable={stable_on} (init={initial_on:.2e}, final={final_on:.2e}); OFF: explodes={explodes_off} (init={initial_off:.2e}, final={final_off:.2e})"

        metrics = {'E_highk_on': E_highk_on, 'E_highk_off': E_highk_off}

        return {'passed': passed, 'metrics': metrics, 'diagnosis': diagnosis}