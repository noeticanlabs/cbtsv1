import numpy as np
import logging
import copy
from src.core.gr_solver import GRSolver

class TestGcatGr4:
    """
    Boundary coherence adversary: Perturb boundary values,
    evolve, and check if boundary perturbations propagate
    or cause global coherence loss (high eps_H).
    """
    def __init__(self, gr_solver: GRSolver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        # Initialize Minkowski
        self.gr_solver.init_minkowski()

        # Perturb boundary: set boundary of gamma_sym6 to different value
        N = self.gr_solver.fields.Nx
        amp = 1e-6
        # Assume periodic, but perturb one face
        self.gr_solver.fields.gamma_sym6[0, :, :, 0] += amp  # x=0 face, xx component

        # Reset time
        self.gr_solver.orchestrator.t = 0.0
        self.gr_solver.orchestrator.step = 0

        # Evolve for a few steps
        max_eps_H = 0.0
        for step in range(5):
            dt, _, _ = self.gr_solver.orchestrator.run_step()
            self.gr_solver.constraints.compute_hamiltonian()
            max_eps_H = max(max_eps_H, self.gr_solver.constraints.eps_H)

        # Check if boundary perturbation causes coherence loss: eps_H > 1e-10
        coherence_lost = max_eps_H > 1e-10
        passed = coherence_lost  # Adversary succeeds if coherence lost

        metrics = {
            'max_eps_H': max_eps_H,
            'amp': amp
        }
        diagnosis = f"Adversary succeeded: boundary coherence lost, eps_H={max_eps_H:.2e}" if passed else f"Adversary failed: boundary coherent, eps_H={max_eps_H:.2e}"

        return {'passed': passed, 'metrics': metrics, 'diagnosis': diagnosis}

if __name__ == "__main__":
    import json
    logging.getLogger().setLevel(logging.CRITICAL)
    solver = GRSolver(16, 16, 16, dx=1.0, dy=1.0, dz=1.0)
    result = TestGcatGr4(solver).run()
    print(result)
    with open('receipts_gcat_gr_4.json', 'w') as f:
        json.dump(result, f, indent=2)