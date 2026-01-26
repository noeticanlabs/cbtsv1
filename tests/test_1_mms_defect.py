"""
Simplified MMS defect test - evaluates RHS with sources applied and checks defect.
"""
import numpy as np
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gr_solver.gr_solver import GRSolver
from gr_solver.gr_core_fields import det_sym6, inv_sym6, sym6_to_mat33, trace_sym6
from tests.gr_test_utils import estimate_order

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MmsDefectTest:
    """Test that sources correctly cancel the RHS defect."""
    
    def __init__(self, N=16, L=16.0):
        self.N = N
        self.L = L
        self.dx = L / N
        
    def set_mms(self, t):
        """Set MMS solution at time t."""
        x = np.arange(self.N) * self.dx
        y = np.arange(self.N) * self.dx
        z = np.arange(self.N) * self.dx
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        kx = 2.0 * np.pi / self.L
        ky = 2.0 * np.pi / self.L
        kz = 2.0 * np.pi / self.L
        kdotx = kx * X + ky * Y + kz * Z
        omega = 1.0
        eps = 1e-5
        
        gamma = np.zeros((self.N, self.N, self.N, 6))
        gamma[..., 0] = 1 + eps * np.sin(kdotx) * np.sin(omega * t)  # xx
        gamma[..., 1] = eps * np.cos(kdotx) * np.cos(omega * t)      # xy
        gamma[..., 3] = 1 + eps * np.sin(kdotx) * np.sin(omega * t)  # yy
        gamma[..., 4] = eps * np.sin(kdotx) * np.sin(omega * t)      # yz
        gamma[..., 5] = 1 + eps * np.cos(kdotx) * np.cos(omega * t)  # zz
        
        K = np.zeros((self.N, self.N, self.N, 6))
        K[..., 0] = eps * np.cos(kdotx) * np.sin(omega * t)
        K[..., 1] = eps * np.sin(kdotx) * np.cos(omega * t)
        K[..., 3] = eps * np.cos(kdotx) * np.sin(omega * t)
        K[..., 4] = eps * np.sin(kdotx) * np.sin(omega * t)
        K[..., 5] = eps * np.cos(kdotx) * np.sin(omega * t)
        
        alpha = np.ones((self.N, self.N, self.N))
        beta = np.zeros((self.N, self.N, self.N, 3))
        
        return gamma, K, alpha, beta
    
    def compute_dt_mms(self, t):
        """Compute exact time derivatives."""
        x = np.arange(self.N) * self.dx
        y = np.arange(self.N) * self.dx
        z = np.arange(self.N) * self.dx
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        kx = 2.0 * np.pi / self.L
        ky = 2.0 * np.pi / self.L
        kz = 2.0 * np.pi / self.L
        kdotx = kx * X + ky * Y + kz * Z
        omega = 1.0
        eps = 1e-5
        
        dt_gamma = np.zeros((self.N, self.N, self.N, 6))
        dt_gamma[..., 0] = eps * omega * np.cos(omega * t) * np.sin(kdotx)
        dt_gamma[..., 1] = -eps * omega * np.cos(kdotx) * np.sin(omega * t)
        dt_gamma[..., 3] = eps * omega * np.cos(omega * t) * np.sin(kdotx)
        dt_gamma[..., 4] = eps * omega * np.sin(kdotx) * np.cos(omega * t)
        dt_gamma[..., 5] = -eps * omega * np.cos(kdotx) * np.sin(omega * t)
        
        dt_K = np.zeros((self.N, self.N, self.N, 6))
        dt_K[..., 0] = eps * omega * np.cos(omega * t) * np.cos(kdotx)
        dt_K[..., 1] = eps * omega * np.sin(kdotx) * (-np.sin(omega * t))
        dt_K[..., 3] = eps * omega * np.cos(omega * t) * np.cos(kdotx)
        dt_K[..., 4] = eps * omega * np.sin(kdotx) * (-np.cos(omega * t))
        dt_K[..., 5] = eps * omega * np.cos(omega * t) * np.cos(kdotx)
        
        return dt_gamma, dt_K
    
    def compute_bssn_dt(self, gamma, K, dt_gamma, dt_K):
        """Compute BSSN time derivatives from physical ones."""
        gamma_inv = inv_sym6(gamma)
        
        # dt_phi = 1/12 * tr(dt_gamma * gamma_inv)
        tr_dt_gamma = trace_sym6(dt_gamma, gamma_inv)
        dt_phi = (1.0/12.0) * tr_dt_gamma
        
        # dt_gamma_tilde = -4 dt_phi * gamma_tilde + e^-4phi * dt_gamma
        det_g = det_sym6(gamma)
        phi = (1.0/12.0) * np.log(det_g)
        exp_minus_4phi = np.exp(-4.0 * phi)[..., np.newaxis]
        gamma_tilde = gamma * exp_minus_4phi
        dt_gamma_tilde = -4.0 * dt_phi[..., np.newaxis] * gamma_tilde + exp_minus_4phi * dt_gamma
        
        # dt_K_trace
        K_mat = sym6_to_mat33(K)
        dt_K_mat = sym6_to_mat33(dt_K)
        gamma_mat = sym6_to_mat33(gamma)
        gamma_inv_mat = sym6_to_mat33(gamma_inv)
        dt_gamma_mat = sym6_to_mat33(dt_gamma)
        dt_gamma_inv_mat = - np.einsum('...ij,...jk,...kl->...il', gamma_inv_mat, dt_gamma_mat, gamma_inv_mat)
        dt_K_trace = np.einsum('...ij,...ji->...', dt_gamma_inv_mat, K_mat) + np.einsum('...ij,...ji->...', gamma_inv_mat, dt_K_mat)
        
        # dt_A_tilde
        K_trace = trace_sym6(K, gamma_inv)
        A_physical = K - (1.0/3.0) * gamma * K_trace[..., np.newaxis]
        A_tilde = A_physical * exp_minus_4phi
        term_brackets = dt_K - (1.0/3.0) * (dt_gamma * K_trace[..., np.newaxis] + gamma * dt_K_trace[..., np.newaxis])
        dt_A_tilde = -4.0 * dt_phi[..., np.newaxis] * A_tilde + exp_minus_4phi * term_brackets
        
        return dt_phi, dt_gamma_tilde, dt_K_trace, dt_A_tilde
    
    def run(self):
        """Run defect test at t=0."""
        solver = GRSolver(self.N, self.N, self.N, dx=self.dx, dy=self.dx, dz=self.dx)
        
        # Set MMS solution
        gamma, K, alpha, beta = self.set_mms(0.0)
        solver.fields.gamma_sym6[:] = gamma
        solver.fields.K_sym6[:] = K
        solver.fields.alpha[:] = alpha
        solver.fields.beta[:] = beta
        
        # Initialize BSSN variables
        det_g = det_sym6(gamma)
        phi = (1.0/12.0) * np.log(det_g)
        solver.fields.phi[:] = phi
        exp_minus_4phi = np.exp(-4.0 * phi)[..., np.newaxis]
        solver.fields.gamma_tilde_sym6[:] = gamma * exp_minus_4phi
        
        gamma_inv = inv_sym6(gamma)
        K_trace = trace_sym6(K, gamma_inv)
        solver.fields.K_trace[:] = K_trace
        
        A_physical = K - (1.0/3.0) * gamma * K_trace[..., np.newaxis]
        solver.fields.A_sym6[:] = A_physical * exp_minus_4phi
        
        solver.fields.Gamma_tilde[:] = 0.0
        solver.fields.Z[:] = 0.0
        solver.fields.Z_i[:] = 0.0
        
        # Compute geometry
        solver.geometry.compute_all()
        
        # Compute RHS without sources
        t = 0.0
        solver.stepper.rhs_computer.sources_func = None
        solver.stepper.rhs_computer.compute_rhs(t, slow_update=True)
        rhs = solver.stepper.rhs_computer
        
        # Compute exact time derivatives
        dt_gamma, dt_K = self.compute_dt_mms(t)
        dt_phi, dt_gamma_tilde, dt_K_trace, dt_A_tilde = self.compute_bssn_dt(gamma, K, dt_gamma, dt_K)
        
        # Compute defects (without sources)
        defect_gamma = rhs.rhs_gamma_sym6 - dt_gamma
        defect_K = rhs.rhs_K_sym6 - dt_K
        defect_gamma_tilde = rhs.rhs_gamma_tilde_sym6 - dt_gamma_tilde
        defect_A = rhs.rhs_A_sym6 - dt_A_tilde
        defect_phi = rhs.rhs_phi - dt_phi
        
        logger.info("=== DEFECT WITHOUT SOURCES ===")
        logger.info(f"||dt_gamma - RHS_gamma||_inf = {np.max(np.abs(defect_gamma)):.6e}")
        logger.info(f"||dt_K - RHS_K||_inf = {np.max(np.abs(defect_K)):.6e}")
        logger.info(f"||dt_gamma_tilde - RHS_gamma_tilde||_inf = {np.max(np.abs(defect_gamma_tilde)):.6e}")
        logger.info(f"||dt_A - RHS_A||_inf = {np.max(np.abs(defect_A)):.6e}")
        logger.info(f"||dt_phi - RHS_phi||_inf = {np.max(np.abs(defect_phi)):.6e}")
        
        # Compute sources that would cancel defect
        sources = {
            'S_gamma_sym6': -defect_gamma,
            'S_K_sym6': -defect_K,
            'S_gamma_tilde_sym6': -defect_gamma_tilde,
            'S_A_sym6': -defect_A,
            'S_phi': -defect_phi,
            'S_Gamma_tilde': -rhs.rhs_Gamma_tilde,  # dt_Gamma_tilde = 0
            'S_Z': -rhs.rhs_Z,
            'S_Z_i': -rhs.rhs_Z_i
        }
        
        # Apply sources and recompute RHS
        solver.stepper.rhs_computer.sources_func = lambda t: sources
        solver.stepper.rhs_computer.compute_rhs(t, slow_update=True)
        rhs_with = solver.stepper.rhs_computer
        
        # Check residual with sources
        residual_gamma = rhs_with.rhs_gamma_sym6 - dt_gamma
        residual_K = rhs_with.rhs_K_sym6 - dt_K
        residual_gamma_tilde = rhs_with.rhs_gamma_tilde_sym6 - dt_gamma_tilde
        residual_A = rhs_with.rhs_A_sym6 - dt_A_tilde
        residual_phi = rhs_with.rhs_phi - dt_phi
        
        logger.info("=== RESIDUAL WITH SOURCES ===")
        logger.info(f"||dt_gamma - RHS_gamma||_inf = {np.max(np.abs(residual_gamma)):.6e}")
        logger.info(f"||dt_K - RHS_K||_inf = {np.max(np.abs(residual_K)):.6e}")
        logger.info(f"||dt_gamma_tilde - RHS_gamma_tilde||_inf = {np.max(np.abs(residual_gamma_tilde)):.6e}")
        logger.info(f"||dt_A - RHS_A||_inf = {np.max(np.abs(residual_A)):.6e}")
        logger.info(f"||dt_phi - RHS_phi||_inf = {np.max(np.abs(residual_phi)):.6e}")
        
        # Check if sources are applied
        sources_applied = (
            np.max(np.abs(residual_gamma)) < 1e-10 and
            np.max(np.abs(residual_K)) < 1e-10 and
            np.max(np.abs(residual_gamma_tilde)) < 1e-10
        )
        
        passed = sources_applied
        logger.info(f"\nSources applied correctly: {passed}")
        
        return {
            'passed': bool(passed),
            'defect_gamma': float(np.max(np.abs(defect_gamma))),
            'defect_K': float(np.max(np.abs(defect_K))),
            'residual_gamma': float(np.max(np.abs(residual_gamma))),
            'residual_K': float(np.max(np.abs(residual_K))),
        }


if __name__ == "__main__":
    import json
    test = MmsDefectTest(N=16, L=16.0)
    result = test.run()
    print(json.dumps(result, indent=2))
