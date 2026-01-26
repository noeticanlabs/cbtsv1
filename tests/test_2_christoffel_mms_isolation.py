import numpy as np
import sys
import os
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gr_solver.gr_geometry_nsc import compute_christoffels_compiled
from gr_solver.gr_core_fields import inv_sym6, sym6_to_mat33

def test_christoffel_isolation(print_shapes=False):
    print("Running Christoffel Isolation Test...")
    
    N = 8
    L = 1.0
    dx = L / N
    dy = dx
    dz = dx
    
    # 1. Setup MMS Metric (Analytic)
    x = np.arange(N) * dx
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    # Simple diagonal metric with variation
    gamma_sym6 = np.zeros((N, N, N, 6))
    gamma_sym6[..., 0] = 1.0 + 0.1 * np.sin(2*np.pi*X) # xx
    gamma_sym6[..., 3] = 1.0 + 0.1 * np.cos(2*np.pi*Y) # yy
    gamma_sym6[..., 5] = 1.0                           # zz
    
    # 2. Analytic Derivatives (dgamma_d[k, i, j] = d_k gamma_ij)
    dgamma_dx_sym6 = np.zeros((N, N, N, 6))
    dgamma_dx_sym6[..., 0] = 0.1 * 2*np.pi * np.cos(2*np.pi*X) # d_x gamma_xx
    
    dgamma_dy_sym6 = np.zeros((N, N, N, 6))
    dgamma_dy_sym6[..., 3] = -0.1 * 2*np.pi * np.sin(2*np.pi*Y) # d_y gamma_yy
    
    dgamma_dz_sym6 = np.zeros((N, N, N, 6))
    
    # 3. Call Compiled Routine
    print("Calling compute_christoffels_compiled...")
    christoffels_num, Gamma_num = compute_christoffels_compiled(
        gamma_sym6, dgamma_dx_sym6, dgamma_dy_sym6, dgamma_dz_sym6
    )
    
    if print_shapes:
        print(f"gamma_sym6 shape: {gamma_sym6.shape}")
        print(f"christoffels_num shape: {christoffels_num.shape}")
        print(f"Gamma_num shape: {Gamma_num.shape}")
        
    # 4. Compute Reference (Python, slow, explicit)
    print("Computing reference...")
    gamma_full = sym6_to_mat33(gamma_sym6)
    gamma_inv = np.linalg.inv(gamma_full) # (N,N,N,3,3)
    
    # Construct d_k gamma_ij
    dgamma = np.zeros((N, N, N, 3, 3, 3)) # k, i, j
    
    # Fill dgamma from analytic expressions
    # k=0 (x)
    dgamma[..., 0, 0, 0] = dgamma_dx_sym6[..., 0]
    # k=1 (y)
    dgamma[..., 1, 1, 1] = dgamma_dy_sym6[..., 3]
    
    # Christoffel symbols of the second kind: Gamma^i_{jk}
    # Gamma^i_{jk} = 0.5 * g^il * (d_j g_lk + d_k g_lj - d_l g_jk)
    christoffels_ref = np.zeros((N, N, N, 3, 3, 3)) # i, j, k
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                val = np.zeros((N, N, N))
                for l in range(3):
                    # term = d_j g_lk + d_k g_lj - d_l g_jk
                    # dgamma indices are (deriv_dir, metric_row, metric_col)
                    term = dgamma[..., j, l, k] + dgamma[..., k, l, j] - dgamma[..., l, j, k]
                    val += 0.5 * gamma_inv[..., i, l] * term
                christoffels_ref[..., i, j, k] = val

    # 5. Compare
    diff = np.max(np.abs(christoffels_num - christoffels_ref))
    print(f"Max difference: {diff:.4e}")
    
    if diff > 1e-10:
        print("MISMATCH DETECTED.")
        print("Possible index permutation error in Christoffel assembly.")
    else:
        print("MATCH. Operator appears correct for this metric.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-shapes", action="store_true")
    args = parser.parse_args()
    test_christoffel_isolation(args.print_shapes)