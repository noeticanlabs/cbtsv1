import numpy as np
import logging
import copy
from gr_solver.gr_solver import GRSolver

class TestGcatGr0:
    """
    Flat spacetime preservation: Evolve Minkowski spacetime and ensure
    constraint violations remain at machine precision floor.
    """
    def __init__(self, gr_solver: GRSolver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        # Initialize with exact Minkowski data
        self.gr_solver.init_minkowski()

        # Compute initial constraints
        self.gr_solver.constraints.compute_hamiltonian()
        self.gr_solver.constraints.compute_momentum()
        self.gr_solver.constraints.compute_residuals()
        eps_H_init = self.gr_solver.constraints.eps_H
        eps_M_init = self.gr_solver.constraints.eps_M

        # Reset time and step
        self.gr_solver.orchestrator.t = 0.0
        self.gr_solver.orchestrator.step = 0

        # Evolve for 10 steps
        max_eps_H = eps_H_init
        max_eps_M = eps_M_init
        for step in range(10):
            dt, _, _ = self.gr_solver.orchestrator.run_step()
            self.gr_solver.constraints.compute_residuals()
            max_eps_H = max(max_eps_H, self.gr_solver.constraints.eps_H)
            max_eps_M = max(max_eps_M, self.gr_solver.constraints.eps_M)

        # Check that max eps_H and eps_M are still at floor (e.g., < 1e-12)
        floor = 1e-12
        passed = (max_eps_H < floor) and (max_eps_M < floor)

        metrics = {
            'eps_H_init': eps_H_init,
            'eps_M_init': eps_M_init,
            'max_eps_H': max_eps_H,
            'max_eps_M': max_eps_M,
            'floor': floor
        }
        diagnosis = f"Preservation passed: max_eps_H={max_eps_H:.2e} < {floor}, max_eps_M={max_eps_M:.2e} < {floor}" if passed else f"Preservation failed: max_eps_H={max_eps_H:.2e} >= {floor} or max_eps_M={max_eps_M:.2e} >= {floor}"

        return {'passed': passed, 'metrics': metrics, 'diagnosis': diagnosis}

if __name__ == "__main__":
    import json
    logging.getLogger().setLevel(logging.CRITICAL)
    solver = GRSolver(16, 16, 16, dx=1.0, dy=1.0, dz=1.0)
    result = TestGcatGr0(solver).run()
    print(result)
    with open('receipts_gcat_gr_0.json', 'w') as f:
        json.dump(result, f, indent=2)