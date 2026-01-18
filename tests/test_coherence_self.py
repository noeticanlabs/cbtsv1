import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
from gr_solver.gr_solver import GRSolver
from gr_solver.gr_core_fields import SYM6_IDX

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoherenceSelfTest:
    """
    Self-test for the coherence system: verifies that coherence operators
    maintain bounded constraint violations during evolution.
    """

    def __init__(self, N=16, L=2.0):
        self.N = N
        self.L = L
        self.dx = L / N
        self.solver = GRSolver(Nx=N, Ny=N, Nz=N, dx=self.dx, dy=self.dx, dz=self.dx)
        # Relax rails for test
        self.solver.orchestrator.rails.H_max = 1.0
        self.solver.orchestrator.rails.M_max = 1.0
        # Disable loom substepping
        self.solver.orchestrator.dt_loom_prev = None

    def run(self):
        logger.info("Running Coherence Self-Test...")

        # Initialize with small perturbation
        self.solver.init_minkowski()

        # Add small perturbation to generate dynamics
        amp = 0.01
        x = np.arange(self.N) * self.dx
        y = np.arange(self.N) * self.dx
        z = np.arange(self.N) * self.dx
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        pert = amp * np.sin(2 * np.pi * X / self.L) * np.sin(2 * np.pi * Y / self.L)
        self.solver.fields.alpha += pert
        self.solver.fields.gamma_sym6[..., SYM6_IDX["xx"]] += pert

        # Compute initial constraints
        self.solver.geometry.compute_christoffels()
        self.solver.geometry.compute_ricci()
        self.solver.geometry.compute_scalar_curvature()
        self.solver.constraints.compute_hamiltonian()
        self.solver.constraints.compute_momentum()
        self.solver.constraints.compute_residuals()

        eps_H_init = self.solver.constraints.eps_H
        eps_M_init = self.solver.constraints.eps_M
        logger.info(f"Initial eps_H: {eps_H_init:.2e}, eps_M: {eps_M_init:.2e}")

        # Run short evolution
        num_steps = 5
        eps_H_max = 0.0
        eps_M_max = 0.0

        for step in range(num_steps):
            dt, dominant_thread, rail_violation = self.solver.orchestrator.run_step(dt_max=0.01)
            # Update max residuals
            eps_H_max = max(eps_H_max, self.solver.constraints.eps_H)
            eps_M_max = max(eps_M_max, self.solver.constraints.eps_M)
            logger.info(f"Step {step}: eps_H={self.solver.constraints.eps_H:.2e}, eps_M={self.solver.constraints.eps_M:.2e}, dominant={dominant_thread}")

        logger.info(f"Max eps_H: {eps_H_max:.2e}, Max eps_M: {eps_M_max:.2e}")

        # Check coherence: residuals should not explode (stay bounded)
        threshold_H = 10.0  # Allow some growth
        threshold_M = 1.0

        success = eps_H_max < threshold_H and eps_M_max < threshold_M
        if success:
            logger.info("Coherence self-test passed: residuals bounded.")
        else:
            logger.error(f"Coherence failed: eps_H_max={eps_H_max:.2e}, eps_M_max={eps_M_max:.2e}")

        return success

if __name__ == "__main__":
    test = CoherenceSelfTest()
    passed = test.run()
    print(f"Coherence Self-Test Passed: {passed}")