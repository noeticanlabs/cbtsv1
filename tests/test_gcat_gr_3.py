import numpy as np
import logging
import copy
from gr_solver.gr_solver import GRSolver

class TestGcatGr3:
    """
    Gauge stiffness adversary: Perturb gauge fields (alpha, beta),
    evolve, and check if they quickly relax back to Minkowski values.
    For stiff gauge, perturbations should decay rapidly.
    """
    def __init__(self, gr_solver: GRSolver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        # Initialize Minkowski
        self.gr_solver.init_minkowski()

        # Store initial gauge
        alpha_init = self.gr_solver.fields.alpha.copy()
        beta_init = self.gr_solver.fields.beta.copy()

        # Perturb gauge: add sinusoidal perturbation
        N = self.gr_solver.fields.Nx
        x = np.arange(N) * self.gr_solver.fields.dx
        y = np.arange(N) * self.gr_solver.fields.dy
        z = np.arange(N) * self.gr_solver.fields.dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        pert = 0.1 * np.sin(2 * np.pi * X / N)
        self.gr_solver.fields.alpha += pert
        self.gr_solver.fields.beta[..., 0] += pert

        # Reset time
        self.gr_solver.orchestrator.t = 0.0
        self.gr_solver.orchestrator.step = 0

        # Evolve for several steps
        alpha_max_dev = []
        beta_max_dev = []
        for step in range(10):
            dt, _, _ = self.gr_solver.orchestrator.run_step()
            # Compute deviation from Minkowski
            dev_alpha = np.max(np.abs(self.gr_solver.fields.alpha - 1.0))
            dev_beta = np.max(np.abs(self.gr_solver.fields.beta))
            alpha_max_dev.append(dev_alpha)
            beta_max_dev.append(dev_beta)

        # Check if deviations decay rapidly (stiff gauge)
        initial_dev = alpha_max_dev[0]
        final_dev = alpha_max_dev[-1] + beta_max_dev[-1]
        decayed = final_dev < initial_dev * 0.1  # Decay to <10% of initial

        passed = decayed

        metrics = {
            'initial_dev': initial_dev,
            'final_dev': final_dev,
            'alpha_max_dev': alpha_max_dev,
            'beta_max_dev': beta_max_dev
        }
        diagnosis = f"Gauge stiff: perturbation decayed from {initial_dev:.2e} to {final_dev:.2e}" if passed else f"Gauge not stiff: perturbation persisted or grew to {final_dev:.2e}"

        return {'passed': passed, 'metrics': metrics, 'diagnosis': diagnosis}

if __name__ == "__main__":
    import json
    logging.getLogger().setLevel(logging.CRITICAL)
    solver = GRSolver(16, 16, 16, dx=1.0, dy=1.0, dz=1.0)
    result = TestGcatGr3(solver).run()
    print(result)
    with open('receipts_gcat_gr_3.json', 'w') as f:
        json.dump(result, f, indent=2)