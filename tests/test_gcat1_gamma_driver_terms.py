import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gr_solver.gr_solver import GRSolver
from tests.test_gcat1_calibration_suite import Test1MmsLite

def test_gamma_driver_shift_terms():
    """
    Verify the shift terms in the Gamma driver RHS:
    RHS^i ~ gamma^jk dj dk beta^i + 1/3 gamma^ij dj dk beta^k
    """
    N = 16
    L = 2.0 * np.pi
    dx = L / N
    solver = GRSolver(N, N, N, dx=dx, dy=dx, dz=dx)
    solver.init_minkowski()
    
    # Set up beta: beta^x = sin(x), others 0
    x = np.arange(N) * dx
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    solver.fields.beta[..., 0] = np.sin(X)
    
    # Gamma_tilde is 0 (flat metric), A_tilde is 0, phi is 0, alpha is 1
    # So advection, twist, compression, and alpha terms are zero.
    # Remaining terms: lap_shift + 1/3 grad_div_shift
    
    # Analytical expectation:
    # lap_shift^x = d_xx beta^x = -sin(x)
    # div_beta = d_x beta^x = cos(x)
    # grad_div_shift^x = d_x (cos(x)) = -sin(x)
    # Total RHS^x = -sin(x) + (1/3)*(-sin(x)) = -4/3 sin(x)
    
    expected_x = - (4.0/3.0) * np.sin(X)
    
    # Instantiate test class to access the method
    test_instance = Test1MmsLite(solver)
    
    # Compute RHS
    rhs = test_instance.compute_full_gamma_driver_rhs(solver)
    
    # Check error (finite difference approximation vs analytic)
    err = np.max(np.abs(rhs[..., 0] - expected_x))
    print(f"Max error in Gamma^x RHS: {err:.2e}")
    
    assert err < 0.1, f"Gamma driver shift terms incorrect, error {err:.2e} too large"

if __name__ == "__main__":
    test_gamma_driver_shift_terms()
    print("Gamma driver shift terms verified.")