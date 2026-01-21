#!/usr/bin/env python3
"""
Debug script for NSC system: compile, load, call rhs.
"""
import numpy as np
from nsc_runtime_min import load_nscir, make_rhs_callable

def test_nsc():
    print("Loading IR...")
    ir = load_nscir("gr_rhs.nscir.json")
    print("IR loaded:", ir['module'])

    print("Making callable...")
    rhs = make_rhs_callable("gr_rhs.nscir.json")
    print("Callable made.")

    # Mock fields
    Nx, Ny, Nz = 8, 8, 8
    fields = {
        'gamma_sym6': np.random.randn(Nx, Ny, Nz, 6),
        'K_sym6': np.random.randn(Nx, Ny, Nz, 6),
        'alpha': np.random.randn(Nx, Ny, Nz),
        'beta': np.random.randn(Nx, Ny, Nz, 3),
        'phi': np.random.randn(Nx, Ny, Nz),
        'gamma_tilde_sym6': np.random.randn(Nx, Ny, Nz, 6),
        'A_sym6': np.random.randn(Nx, Ny, Nz, 6),
        'Gamma_tilde': np.random.randn(Nx, Ny, Nz, 3),
        'Z': np.random.randn(Nx, Ny, Nz),
        'Z_i': np.random.randn(Nx, Ny, Nz, 3),
        'dx': 1.0, 'dy': 1.0, 'dz': 1.0
    }

    print("Calling rhs...")
    try:
        result = rhs(fields, lambda_val=0.0, sources_enabled=False)
        print("RHS executed successfully.")
        print("Output keys:", list(result.keys()))
        for k, v in result.items():
            print(f"{k}: shape {v.shape}, mean {v.mean():.3f}")
    except Exception as e:
        print("Error:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_nsc()