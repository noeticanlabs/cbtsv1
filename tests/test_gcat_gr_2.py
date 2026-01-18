import numpy as np
import logging
import copy
from gr_solver.gr_solver import GRSolver

class TestGcatGr2:
    """
    Stage mismatch adversary: Inject a sharp perturbation that causes
    RK stages to see inconsistent RHS, leading to instability.
    Expect gates to fail (eps_H spikes).
    """
    def __init__(self, gr_solver: GRSolver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        # Initialize Minkowski
        self.gr_solver.init_minkowski()

        # Inject sharp perturbation at center (delta-like)
        N = self.gr_solver.fields.Nx
        center = N // 2
        amp = 1e-6
        self.gr_solver.fields.K_sym6[center, center, center, 0] += amp

        # Reset time
        self.gr_solver.orchestrator.t = 0.0
        self.gr_solver.orchestrator.step = 0

        # Evolve for a few steps
        max_eps_H = 0.0
        for step in range(5):
            dt, _, _ = self.gr_solver.orchestrator.run_step()
            self.gr_solver.constraints.compute_hamiltonian()
            max_eps_H = max(max_eps_H, self.gr_solver.constraints.eps_H)
            # If eps_H spikes high, gates fail
            if self.gr_solver.constraints.eps_H > 1e-8:
                break

        # Expect gates to fail: eps_H > 1e-9
        gates_fail = max_eps_H > 1e-9
        passed = gates_fail  # Adversary succeeds if gates fail

        metrics = {
            'max_eps_H': max_eps_H,
            'amp': amp
        }
        diagnosis = f"Adversary succeeded: gates failed with eps_H={max_eps_H:.2e}" if passed else f"Adversary failed: gates held with eps_H={max_eps_H:.2e}"

        return {'passed': passed, 'metrics': metrics, 'diagnosis': diagnosis}

if __name__ == "__main__":
    import json
    logging.getLogger().setLevel(logging.CRITICAL)
    solver = GRSolver(16, 16, 16, dx=1.0, dy=1.0, dz=1.0)
    result = TestGcatGr2(solver).run()
    print(result)
    with open('receipts_gcat_gr_2.json', 'w') as f:
        json.dump(result, f, indent=2)