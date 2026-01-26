#!/usr/bin/env python3
"""
wp1_test4_ricci_spectral_reference.py
-------------------------------------
WP1 / GR calibration: Ricci operator vs spectral (FFT) reference.

Goal:
- Verify convergence of the numerical Ricci tensor computation (R_ij)
  against a high-precision spectral reference on a periodic domain.
- This isolates the Ricci assembly step from time integration.

Method:
1. Construct a smooth periodic metric gamma_ij(x) using low Fourier modes.
2. Compute R_ij using the solver's finite-difference kernel for the raw metric.
3. Compute R_ij_ref using FFT-based derivatives (spectral accuracy).
4. Compare L2 norms of the difference at resolutions N=16, 32, 48.
5. Check for ~2nd order convergence (p_obs ~ 2.0).

Usage:
  python3 tests/wp1_test4_ricci_spectral_reference.py --Ns 16,32,48 --L 16 --eps 1e-3
"""

import numpy as np
import argparse
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gr_solver.gr_solver import GRSolver
from gr_solver.gr_core_fields import sym6_to_mat33, mat33_to_sym6
from gr_solver.gr_geometry_nsc import IDX_XX, IDX_XY, IDX_XZ, IDX_YY, IDX_YZ, IDX_ZZ

def make_grid(N, L):
    dx = L / N
    x = np.arange(N) * dx
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    return X, Y, Z, dx

def build_periodic_metric(N, L, eps, m):
    X, Y, Z, dx = make_grid(N, L)
    k = 2.0 * np.pi * m / L
    
    gamma = np.zeros((N, N, N, 6))
    gamma[..., IDX_XX] = 1.0
    gamma[..., IDX_YY] = 1.0
    gamma[..., IDX_ZZ] = 1.0
    
    gamma[..., IDX_XX] += eps * np.sin(k*X) * np.cos(k*Y)
    gamma[..., IDX_YY] += eps * np.cos(k*X) * np.sin(k*Z)
    gamma[..., IDX_ZZ] += eps * np.sin(k*Y) * np.cos(k*Z)
    
    gamma[..., IDX_XY] += eps * 0.5 * np.sin(k*X) * np.sin(k*Y)
    gamma[..., IDX_XZ] += eps * 0.5 * np.cos(k*X) * np.cos(k*Z)
    gamma[..., IDX_YZ] += eps * 0.5 * np.sin(k*Y) * np.sin(k*Z)
    
    return gamma, dx

def fft_deriv(f, L, axis):
    N = f.shape[axis]
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L/N)
    
    shape = [1] * f.ndim
    shape[axis] = N
    k = k.reshape(shape)
    
    f_hat = np.fft.fftn(f)
    df_hat = 1j * k * f_hat
    df = np.fft.ifftn(df_hat).real
    return df

def sym6_to_mat33_local(sym6):
    shape = sym6.shape[:-1]
    mat = np.zeros(shape + (3, 3), dtype=sym6.dtype)
    mat[..., 0, 0] = sym6[..., IDX_XX]
    mat[..., 0, 1] = sym6[..., IDX_XY]
    mat[..., 0, 2] = sym6[..., IDX_XZ]
    mat[..., 1, 0] = sym6[..., IDX_XY]
    mat[..., 1, 1] = sym6[..., IDX_YY]
    mat[..., 1, 2] = sym6[..., IDX_YZ]
    mat[..., 2, 0] = sym6[..., IDX_XZ]
    mat[..., 2, 1] = sym6[..., IDX_YZ]
    mat[..., 2, 2] = sym6[..., IDX_ZZ]
    return mat

def compute_ricci_spectral(gamma_sym6, L):
    N = gamma_sym6.shape[0]
    g = sym6_to_mat33_local(gamma_sym6)
    g_inv = np.linalg.inv(g)
    
    dg = np.zeros(g.shape + (3,))
    for i in range(3):
        for j in range(3):
            dg[..., i, j, 0] = fft_deriv(g[..., i, j], L, 0)
            dg[..., i, j, 1] = fft_deriv(g[..., i, j], L, 1)
            dg[..., i, j, 2] = fft_deriv(g[..., i, j], L, 2)
            
    Gamma = np.zeros(g.shape + (3,))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                val = np.zeros_like(g[..., 0, 0])
                for l in range(3):
                    term = dg[..., l, k, j] + dg[..., l, j, k] - dg[..., j, k, l]
                    val += 0.5 * g_inv[..., i, l] * term
                Gamma[..., i, j, k] = val
                
    dGamma = np.zeros(Gamma.shape + (3,))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                dGamma[..., i, j, k, 0] = fft_deriv(Gamma[..., i, j, k], L, 0)
                dGamma[..., i, j, k, 1] = fft_deriv(Gamma[..., i, j, k], L, 1)
                dGamma[..., i, j, k, 2] = fft_deriv(Gamma[..., i, j, k], L, 2)
                
    Ricci = np.zeros(g.shape)
    for i in range(3):
        for j in range(3):
            term1 = np.zeros_like(g[..., 0, 0])
            term2 = np.zeros_like(g[..., 0, 0])
            term3 = np.zeros_like(g[..., 0, 0])
            term4 = np.zeros_like(g[..., 0, 0])
            
            for k in range(3):
                term1 += dGamma[..., k, i, j, k]
                term2 += dGamma[..., k, i, k, j]
                for l in range(3):
                    term3 += Gamma[..., k, k, l] * Gamma[..., l, i, j]
                    term4 += Gamma[..., k, j, l] * Gamma[..., l, i, k]
            
            Ricci[..., i, j] = term1 - term2 + term3 - term4
            
    return Ricci

def l2_norm(f):
    return np.sqrt(np.mean(f**2))

def estimate_order(errors, hs):
    if len(errors) < 2:
        return 0.0
    p = np.log(errors[-2] / errors[-1]) / np.log(hs[-2] / hs[-1])
    return float(p)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ns", type=str, default="16,32,48")
    parser.add_argument("--L", type=float, default=16.0)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--crop", type=int, default=0)
    parser.add_argument("--p-min", type=float, default=1.8)
    args = parser.parse_args()
    
    Ns = [int(x) for x in args.Ns.split(',')]
    errors = []
    hs = []
    
    print(f"Running Ricci Spectral Reference Test")
    print(f"L={args.L}, eps={args.eps}, m={args.m}, crop={args.crop}")
    
    for N in Ns:
        dx = args.L / N
        print(f"\nTesting N={N}, dx={dx:.4f}...")
        
        solver = GRSolver(N, N, N, dx=dx, dy=dx, dz=dx)
        
        gamma_sym6, _ = build_periodic_metric(N, args.L, args.eps, args.m)
        solver.fields.gamma_sym6 = gamma_sym6.copy()

        # This test is for the RAW Ricci operator, not the BSSN one.
        # We call compute_ricci_for_metric directly.
        solver.geometry.clear_cache()
        solver.geometry.compute_christoffels()
        R_solver = solver.geometry.compute_ricci_for_metric(solver.fields.gamma_sym6,
                                                          solver.geometry.christoffels)
        
        R_ref = compute_ricci_spectral(gamma_sym6, args.L)
        
        diff = R_solver - R_ref
        
        if args.crop > 0:
            c = args.crop
            diff = diff[c:-c, c:-c, c:-c]
            
        error = l2_norm(diff)
        errors.append(error)
        hs.append(dx)
        
        print(f"  Error L2: {error:.4e}")
        
    p_obs = estimate_order(errors, hs)
    print(f"\nObserved convergence order p_obs = {p_obs:.2f}")
    passed = p_obs >= args.p_min
    
    result = {
        "passed": passed,
        "metrics": {
            "p_obs": p_obs if not np.isnan(p_obs) else None,
            "errors": [float(e) for e in errors],
            "resolutions": Ns
        }
    }
    
    print(json.dumps(result, indent=2))
    
    if not passed:
        sys.exit(1)

if __name__ == "__main__":
    main()
