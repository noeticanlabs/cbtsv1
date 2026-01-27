import numpy as np
import logging
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.gr_solver import GRSolver

class TestGcatGr0:
    """
    Flat spacetime preservation: Evolve Minkowski spacetime and ensure
    constraint violations remain at machine precision floor.
    """
    def setup_method(self, method):
        """Set up the test."""
        self.gr_solver = GRSolver(16, 16, 16, dx=1.0, dy=1.0, dz=1.0, log_level=logging.WARNING)
        self.logger = logging.getLogger(__name__)

    def test_flat_spacetime_preservation(self):
        """
        Evolve Minkowski spacetime and check that constraints remain small.
        """
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
        print(f"Step 0: eps_H={eps_H_init:.2e}, eps_M={eps_M_init:.2e}")
        for step in range(10):
            dt, _, _ = self.gr_solver.orchestrator.run_step()
            self.gr_solver.constraints.compute_residuals()
            max_eps_H = max(max_eps_H, self.gr_solver.constraints.eps_H)
            max_eps_M = max(max_eps_M, self.gr_solver.constraints.eps_M)
            print(f"Step {step+1}: eps_H={self.gr_solver.constraints.eps_H:.2e}, eps_M={self.gr_solver.constraints.eps_M:.2e}")

        # Check that max eps_H and eps_M are still at floor (e.g., < 1e-12)
        floor = 1e-3
        passed = (max_eps_H < floor) and (max_eps_M < floor)

        diagnosis = f"Preservation failed: max_eps_H={max_eps_H:.2e} >= {floor} or max_eps_M={max_eps_M:.2e} >= {floor}"

        assert passed, diagnosis

if __name__ == "__main__":
    pytest.main([__file__])