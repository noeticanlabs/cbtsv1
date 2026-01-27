"""
Test 2 RCS
"""

import numpy as np
import logging
from src.core.gr_solver import GRSolver

class Test2Rcs:
    def __init__(self, gr_solver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        N = 8
        resolutions = [N, 2*N, 4*N]
        T = 0.1
        solutions = {}
        for res in resolutions:
            dx = 1.0 / res  # fixed domain [0,1]^3
            solver = GRSolver(res, res, res, dx=dx, dy=dx, dz=dx)
            solver.init_minkowski()
            solver.orchestrator.t = 0.0
            solver.orchestrator.step = 0
            while solver.orchestrator.t < T:
                dt_max = T - solver.orchestrator.t
                dt, _, _ = solver.orchestrator.run_step(dt_max)
            solutions[res] = {
                'gamma_sym6': solver.fields.gamma_sym6.copy(),
                'K_sym6': solver.fields.K_sym6.copy()
            }
        # Now restrict finer to coarser
        def restrict(fine, coarse_res):
            fine_res = 2 * coarse_res
            assert fine.shape[:3] == (fine_res, fine_res, fine_res)
            coarse = np.zeros((coarse_res, coarse_res, coarse_res) + fine.shape[3:])
            for i in range(coarse_res):
                for j in range(coarse_res):
                    for k in range(coarse_res):
                        coarse[i, j, k] = np.mean(fine[2*i:2*i+2, 2*j:2*j+2, 2*k:2*k+2], axis=(0,1,2))
            return coarse
        
        # Compute differences
        gamma_N = solutions[N]['gamma_sym6']
        K_N = solutions[N]['K_sym6']
        gamma_2N = solutions[2*N]['gamma_sym6']
        K_2N = solutions[2*N]['K_sym6']
        gamma_4N = solutions[4*N]['gamma_sym6']
        K_4N = solutions[4*N]['K_sym6']
        
        # Restrict 2N to N
        gamma_2N_restricted = restrict(gamma_2N, N)
        K_2N_restricted = restrict(K_2N, N)
        # Restrict 4N to 2N
        gamma_4N_restricted = restrict(gamma_4N, 2*N)
        K_4N_restricted = restrict(K_4N, 2*N)
        
        # L2 differences
        diff_gamma_coarse = np.sqrt(np.sum((gamma_N - gamma_2N_restricted)**2))
        diff_K_coarse = np.sqrt(np.sum((K_N - K_2N_restricted)**2))
        diff_coarse = diff_gamma_coarse + diff_K_coarse
        
        diff_gamma_fine = np.sqrt(np.sum((gamma_2N - gamma_4N_restricted)**2))
        diff_K_fine = np.sqrt(np.sum((K_2N - K_4N_restricted)**2))
        diff_fine = diff_gamma_fine + diff_K_fine
        
        if diff_fine == 0:
            p_obs = np.inf
        else:
            p_obs = np.log2(diff_coarse / diff_fine)
        
        p_target = 2
        passed = p_obs > p_target - 0.5
        
        metrics = {'p_obs': p_obs, 'diff_coarse': diff_coarse, 'diff_fine': diff_fine}
        diagnosis = f"Observed order p_obs = {p_obs:.2f}, expected ~{p_target}, {'passed' if passed else 'failed'} (threshold {p_target - 0.5})"
        
        return {'passed': passed, 'metrics': metrics, 'diagnosis': diagnosis}