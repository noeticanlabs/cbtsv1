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

class SchwarzschildSetup:
    """Helper to initialize Schwarzschild."""

    def __init__(self, solver):
        self.solver = solver

    def init_schwarzschild(self):
        """Initialize fields to Schwarzschild metric in isotropic coordinates."""
        logger.info("Initializing Schwarzschild metric...")

        # Create centered coordinates
        x = (np.arange(self.solver.fields.Nx) - self.solver.fields.Nx//2) * self.solver.fields.dx
        y = (np.arange(self.solver.fields.Ny) - self.solver.fields.Ny//2) * self.solver.fields.dy
        z = (np.arange(self.solver.fields.Nz) - self.solver.fields.Nz//2) * self.solver.fields.dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        r = np.sqrt(X**2 + Y**2 + Z**2)
        r = np.maximum(r, 0.1)

        # Conformal factor ψ
        psi = np.sqrt(1.0 + 1.0 / (2.0 * r))

        # Spatial metric γ_ij = ψ^4 δ_ij
        gamma_factor = psi**4
        self.solver.fields.gamma_sym6[..., SYM6_IDX["xx"]] = gamma_factor
        self.solver.fields.gamma_sym6[..., SYM6_IDX["yy"]] = gamma_factor
        self.solver.fields.gamma_sym6[..., SYM6_IDX["zz"]] = gamma_factor
        self.solver.fields.gamma_sym6[..., SYM6_IDX["xy"]] = 0.0
        self.solver.fields.gamma_sym6[..., SYM6_IDX["xz"]] = 0.0
        self.solver.fields.gamma_sym6[..., SYM6_IDX["yz"]] = 0.0

        # Lapse α
        alpha_num = 1.0 - 1.0 / (2.0 * r)
        alpha_den = 1.0 + 1.0 / (2.0 * r)
        self.solver.fields.alpha = alpha_num / alpha_den

        # Shift β^i = 0
        self.solver.fields.beta.fill(0.0)

        # Extrinsic curvature K_ij = 0
        self.solver.fields.K_sym6.fill(0.0)

        # BSSN fields
        self.solver.fields.phi = np.log(psi)
        self.solver.fields.gamma_tilde_sym6[..., SYM6_IDX["xx"]] = 1.0
        self.solver.fields.gamma_tilde_sym6[..., SYM6_IDX["yy"]] = 1.0
        self.solver.fields.gamma_tilde_sym6[..., SYM6_IDX["zz"]] = 1.0
        self.solver.fields.gamma_tilde_sym6[..., SYM6_IDX["xy"]] = 0.0
        self.solver.fields.gamma_tilde_sym6[..., SYM6_IDX["xz"]] = 0.0
        self.solver.fields.gamma_tilde_sym6[..., SYM6_IDX["yz"]] = 0.0
        self.solver.fields.A_sym6.fill(0.0)
        self.solver.fields.Gamma_tilde.fill(0.0)
        self.solver.fields.Z.fill(0.0)
        self.solver.fields.Z_i.fill(0.0)

class SolverAccuracyTest:
    """
    Test solver accuracy by comparing numerical constraints to analytical
    expectations for the Schwarzschild vacuum solution.
    Measures L2 norm of H and M residuals.
    """

    def __init__(self, N_list=[8, 16]):
        self.N_list = N_list

    def run(self):
        logger.info("Running Solver Accuracy Test...")

        results = {}

        for N in self.N_list:
            logger.info(f"Testing N={N}...")

            solver = GRSolver(Nx=N, Ny=N, Nz=N, dx=10.0/N, dy=10.0/N, dz=10.0/N)
            setup = SchwarzschildSetup(solver)
            setup.init_schwarzschild()

            # Compute constraints
            solver.geometry.compute_christoffels()
            solver.geometry.compute_ricci()
            solver.geometry.compute_scalar_curvature()
            solver.constraints.compute_hamiltonian()
            solver.constraints.compute_momentum()
            solver.constraints.compute_residuals()

            # Compute L2 norms (RMS)
            eps_H_rms = np.sqrt(np.mean(solver.constraints.H**2))
            eps_M_rms = np.sqrt(np.mean(np.sum(solver.constraints.M**2, axis=-1)))

            results[N] = {'eps_H_rms': eps_H_rms, 'eps_M_rms': eps_M_rms}
            logger.info(f"N={N}: eps_H_rms={eps_H_rms:.2e}, eps_M_rms={eps_M_rms:.2e}")

        # Check convergence (rough)
        if len(results) > 1:
            N1, N2 = sorted(results.keys())[:2]
            H1, H2 = results[N1]['eps_H_rms'], results[N2]['eps_H_rms']
            M1, M2 = results[N1]['eps_M_rms'], results[N2]['eps_M_rms']

            # Expect some improvement
            improvement_H = H1 / H2 if H2 > 0 else 0
            improvement_M = M1 / M2 if M2 > 0 else 0

            logger.info(f"Convergence H: {improvement_H:.1f}x, M: {improvement_M:.1f}x")

        # Accuracy thresholds (empirical for Schwarzschild, allow discretization errors)
        threshold_H = 5.0  # L2 norm threshold
        threshold_M = 1e-3

        success = all(r['eps_H_rms'] < threshold_H and r['eps_M_rms'] < threshold_M for r in results.values())
        if success:
            logger.info("Solver accuracy test passed.")
        else:
            logger.error("Solver accuracy test failed: residuals too high.")

        return success

if __name__ == "__main__":
    test = SolverAccuracyTest()
    passed = test.run()
    print(f"Solver Accuracy Test Passed: {passed}")