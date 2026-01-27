"""
Test 5 Gauge Invariance
"""

import numpy as np
import logging
import copy
from src.core.gr_core_fields import inv_sym6, trace_sym6, sym6_to_mat33

class Test5:
    def __init__(self, gr_solver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        T = 0.05
        tol_gamma = 1e-6
        tol_inv = 1e-10

        # Run A: Normal evolution with Minkowski initial data
        solver_A = copy.deepcopy(self.gr_solver)
        solver_A.init_minkowski()
        solver_A.orchestrator.t = 0.0
        solver_A.orchestrator.step = 0
        while solver_A.orchestrator.t < T:
            dt_max = T - solver_A.orchestrator.t
            dt, _, _ = solver_A.orchestrator.run_step(dt_max)

        # Compute invariants for A
        solver_A.geometry.compute_scalar_curvature()
        R_A = solver_A.geometry.R
        gamma_inv_A = inv_sym6(solver_A.fields.gamma_sym6)
        gamma_inv_full_A = sym6_to_mat33(gamma_inv_A)
        K_full_A = sym6_to_mat33(solver_A.fields.K_sym6)
        K_raised_A = np.einsum('...ik,...kj->...ij', gamma_inv_full_A, K_full_A)
        K2_A = np.einsum('...ij,...ij', K_raised_A, K_full_A)
        trK_A = trace_sym6(solver_A.fields.K_sym6, gamma_inv_A)
        gamma_full_A = sym6_to_mat33(solver_A.fields.gamma_sym6)
        det_A = np.linalg.det(gamma_full_A)

        # Run B: Apply gauge scramble
        solver_B = copy.deepcopy(self.gr_solver)
        solver_B.init_minkowski()
        # Perturb alpha and beta with small sinusoidal perturbations
        N = solver_B.fields.N
        eps = 1e-6
        x = np.arange(N) * solver_B.fields.dx
        y = np.arange(N) * solver_B.fields.dy
        z = np.arange(N) * solver_B.fields.dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        pert = eps * np.sin(2 * np.pi * (X + Y + Z) / N)
        solver_B.fields.alpha += pert
        solver_B.fields.beta[..., 0] += pert
        solver_B.fields.beta[..., 1] += pert
        solver_B.fields.beta[..., 2] += pert
        # Evolve
        solver_B.orchestrator.t = 0.0
        solver_B.orchestrator.step = 0
        while solver_B.orchestrator.t < T:
            dt_max = T - solver_B.orchestrator.t
            dt, _, _ = solver_B.orchestrator.run_step(dt_max)

        # Compute invariants for B
        solver_B.geometry.compute_scalar_curvature()
        R_B = solver_B.geometry.R
        gamma_inv_B = inv_sym6(solver_B.fields.gamma_sym6)
        gamma_inv_full_B = sym6_to_mat33(gamma_inv_B)
        K_full_B = sym6_to_mat33(solver_B.fields.K_sym6)
        K_raised_B = np.einsum('...ik,...kj->...ij', gamma_inv_full_B, K_full_B)
        K2_B = np.einsum('...ij,...ij', K_raised_B, K_full_B)
        trK_B = trace_sym6(solver_B.fields.K_sym6, gamma_inv_B)
        gamma_full_B = sym6_to_mat33(solver_B.fields.gamma_sym6)
        det_B = np.linalg.det(gamma_full_B)

        # Compute L2 differences
        diff_gamma = np.sqrt(np.sum((solver_A.fields.gamma_sym6 - solver_B.fields.gamma_sym6)**2))
        diff_R = np.sqrt(np.sum((R_A - R_B)**2))
        diff_K2 = np.sqrt(np.sum((K2_A - K2_B)**2))
        diff_trK = np.sqrt(np.sum((trK_A - trK_B)**2))
        diff_det = np.sqrt(np.sum((det_A - det_B)**2))

        # Pass if gauge fields differ and invariants match
        passed = diff_gamma > tol_gamma and diff_R < tol_inv and diff_K2 < tol_inv and diff_trK < tol_inv and diff_det < tol_inv

        metrics = {'diff_gamma': diff_gamma, 'diff_R': diff_R, 'diff_K2': diff_K2, 'diff_trK': diff_trK, 'diff_det': diff_det}
        diagnosis = f"Gauge scramble applied, gamma differs by {diff_gamma:.2e} (> {tol_gamma}), invariants match within {tol_inv}"

        return {'passed': passed, 'metrics': metrics, 'diagnosis': diagnosis}