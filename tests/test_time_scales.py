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

class TimeScalesTest:
    """
    Test all time scales and their corresponding effects: verifies that the
    GR solver correctly handles perturbations across different spatial/temporal scales.
    Tests low-frequency (slow) and high-frequency (fast) modes.
    """

    def __init__(self):
        self.solver = GRSolver(Nx=16, Ny=16, Nz=16, dx=0.1, dy=0.1, dz=0.1)
        # Relax rails
        self.solver.orchestrator.rails.H_max = 1.0
        self.solver.orchestrator.rails.M_max = 1.0

    def run_scenario(self, k, label):
        logger.info(f"Running {label} scenario (k={k})...")

        # Reset to Minkowski
        self.solver.init_minkowski()

        # Add perturbation with wavenumber k
        x = np.arange(self.solver.fields.Nx) * self.solver.fields.dx
        y = np.arange(self.solver.fields.Ny) * self.solver.fields.dy
        z = np.arange(self.solver.fields.Nz) * self.solver.fields.dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        amp = 0.01
        pert = amp * np.sin(k * X) * np.sin(k * Y) * np.sin(k * Z)
        self.solver.fields.alpha += pert
        self.solver.fields.gamma_sym6[..., SYM6_IDX["xx"]] += pert

        # Compute initial residuals
        self.solver.geometry.compute_christoffels()
        self.solver.geometry.compute_ricci()
        self.solver.geometry.compute_scalar_curvature()
        self.solver.constraints.compute_hamiltonian()
        self.solver.constraints.compute_momentum()
        self.solver.constraints.compute_residuals()

        eps_H_init = self.solver.constraints.eps_H
        eps_M_init = self.solver.constraints.eps_M

        # Run short evolution
        eps_H_max = eps_H_init
        eps_M_max = eps_M_init
        dominants = []

        for step in range(3):
            dt, dominant_thread, rail_violation = self.solver.orchestrator.run_step(dt_max=0.01)
            eps_H_max = max(eps_H_max, self.solver.constraints.eps_H)
            eps_M_max = max(eps_M_max, self.solver.constraints.eps_M)
            dominants.append(dominant_thread)

        logger.info(f"{label}: eps_H_max={eps_H_max:.2e}, eps_M_max={eps_M_max:.2e}, dominants={dominants}")

        return eps_H_max, eps_M_max, dominants

    def run(self):
        logger.info("Running Time Scales Test...")

        # Test low-frequency (k=1, slow scale)
        eps_H_slow, eps_M_slow, dom_slow = self.run_scenario(k=1, label="Low-frequency")

        # Test high-frequency (k=8, fast scale)
        eps_H_fast, eps_M_fast, dom_fast = self.run_scenario(k=8, label="High-frequency")

        # Check effects
        # High-frequency should have higher residuals due to discretization errors
        # But system should handle both

        threshold_H = 1.0
        threshold_M = 0.1

        success_slow = eps_H_slow < threshold_H and eps_M_slow < threshold_M
        success_fast = eps_H_fast < threshold_H and eps_M_fast < threshold_M

        if success_slow and success_fast:
            logger.info("Time scales test passed: both slow and fast scales handled.")
        else:
            logger.error(f"Time scales test failed: slow {success_slow}, fast {success_fast}")

        # Additional check: high-freq should have different dominant threads
        # (may switch to gauge or phys)

        return success_slow and success_fast

if __name__ == "__main__":
    test = TimeScalesTest()
    passed = test.run()
    print(f"Time Scales Test Passed: {passed}")