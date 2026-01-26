import numpy as np
import sys
import os
import logging
import argparse
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gr_solver.gr_solver import GRSolver
from gr_solver.gr_core_fields import SYM6_IDX
from tests.gr_test_utils import estimate_order

class RicciIsolationTest:
    """
    Test 4: Ricci Tensor Isolation
    Verifies the numerical Ricci tensor computation against an analytic reference
    for a conformally flat metric on a periodic domain.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def run(self):
        print("Running Ricci Isolation Test...")
        
        resolutions = [16, 24, 32]
        errors = []
        hs = []
        L = 2.0 * np.pi
        
        for N in resolutions:
            dx = L / N
            solver = GRSolver(N, N, N, dx=dx, dy=dx, dz=dx)
            
            # Setup a non-trivial metric (periodic)
            # conformal factor phi = 0.1 * sin(x)sin(y)sin(z)
            # gamma_ij = e^{4phi} delta_ij
            x = np.arange(N) * dx
            X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
            
            phi = 0.1 * np.sin(X) * np.sin(Y) * np.sin(Z)
            gamma_factor = np.exp(4.0 * phi)
            
            # Initialize metric
            solver.fields.gamma_sym6[...] = 0.0
            solver.fields.gamma_sym6[..., SYM6_IDX["xx"]] = gamma_factor # xx
            solver.fields.gamma_sym6[..., SYM6_IDX["yy"]] = gamma_factor # yy
            solver.fields.gamma_sym6[..., SYM6_IDX["zz"]] = gamma_factor # zz
            
            # Compute numerical Ricci
            solver.geometry.compute_christoffels()
            R_num = solver.geometry.compute_ricci_for_metric(
                solver.fields.gamma_sym6, solver.geometry.christoffels
            )
            
            # Compute analytic Ricci for g_ij = e^(4phi) delta_ij
            # Formula: R_ij = -2 D_i D_j phi - 2 delta_ij lap(phi) + 4 D_i phi D_j phi - 4 delta_ij |grad phi|^2
            # (Note: u = 2phi in standard formula R = ... u ..., here we substitute directly)
            
            dphi_x = 0.1 * np.cos(X) * np.sin(Y) * np.sin(Z)
            dphi_y = 0.1 * np.sin(X) * np.cos(Y) * np.sin(Z)
            dphi_z = 0.1 * np.sin(X) * np.sin(Y) * np.cos(Z)
            
            d2phi_xx = -0.1 * np.sin(X) * np.sin(Y) * np.sin(Z)
            d2phi_yy = -0.1 * np.sin(X) * np.sin(Y) * np.sin(Z)
            d2phi_zz = -0.1 * np.sin(X) * np.sin(Y) * np.sin(Z)
            
            d2phi_xy = 0.1 * np.cos(X) * np.cos(Y) * np.sin(Z)
            d2phi_xz = 0.1 * np.cos(X) * np.sin(Y) * np.cos(Z)
            d2phi_yz = 0.1 * np.sin(X) * np.cos(Y) * np.cos(Z)
            
            lap_phi = d2phi_xx + d2phi_yy + d2phi_zz
            grad_phi_sq = dphi_x**2 + dphi_y**2 + dphi_z**2
            
            R_ref = np.zeros((N, N, N, 3, 3))
            
            # Fill R_ref
            for i, (di, d2ii) in enumerate([(dphi_x, d2phi_xx), (dphi_y, d2phi_yy), (dphi_z, d2phi_zz)]):
                R_ref[..., i, i] = -2*d2ii - 2*lap_phi + 4*di**2 - 4*grad_phi_sq
            
            R_ref[..., 0, 1] = R_ref[..., 1, 0] = -2*d2phi_xy + 4*dphi_x*dphi_y
            R_ref[..., 0, 2] = R_ref[..., 2, 0] = -2*d2phi_xz + 4*dphi_x*dphi_z
            R_ref[..., 1, 2] = R_ref[..., 2, 1] = -2*d2phi_yz + 4*dphi_y*dphi_z
            
            # Compare
            diff = R_num - R_ref
            error = np.sqrt(np.mean(diff**2))
            errors.append(error)
            hs.append(dx)
            
            print(f"  N={N}, Error Ricci={error:.4e}")
            
        p_obs = estimate_order(errors, hs)
        print(f"Observed convergence order p_obs = {p_obs:.2f}")
        
        return {'passed': p_obs > 1.5, 'metrics': {'p_obs': p_obs, 'errors': errors}}

if __name__ == "__main__":
    test = RicciIsolationTest()
    result = test.run()
    print(json.dumps(result, indent=2))