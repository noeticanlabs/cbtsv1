#!/usr/bin/env python3
"""
NSC-compiled constraint computations for GR solver.
Provides JIT-compiled versions of constraint evaluations for performance.
"""

import numpy as np
from typing import Callable, Optional, Dict, Any
try:
    from numba import jit
except ImportError:
    jit = lambda f=None, **kwargs: f if f else (lambda g: g)

# Helper functions (copied from gr_core_fields)
@jit(nopython=True)
def _sym6_to_mat33_jit(sym6):
    mat = np.zeros((3, 3), dtype=sym6.dtype)
    mat[0, 0] = sym6[0]
    mat[0, 1] = sym6[1]
    mat[0, 2] = sym6[2]
    mat[1, 0] = sym6[1]
    mat[1, 1] = sym6[3]
    mat[1, 2] = sym6[4]
    mat[2, 0] = sym6[2]
    mat[2, 1] = sym6[4]
    mat[2, 2] = sym6[5]
    return mat

@jit(nopython=True)
def _inv_sym6_jit(sym6):
    xx, xy, xz, yy, yz, zz = sym6
    det = xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz) + xz * (xy * yz - yy * xz)
    inv = np.empty(6, dtype=sym6.dtype)
    inv[0] = (yy*zz - yz*yz) / det
    inv[1] = -(xy*zz - xz*yz) / det
    inv[2] = (xy*yz - xz*yy) / det
    inv[3] = (xx*zz - xz*xz) / det
    inv[4] = -(xx*yz - xy*xz) / det
    inv[5] = (xx*yy - xy*xy) / det
    return inv

@jit(nopython=True)
def _trace_sym6_jit(sym6, inv_sym6):
    """Compute trace: gamma^{ij} A_ij"""
    return (
        inv_sym6[0]*sym6[0]
      + 2.0*inv_sym6[1]*sym6[1]
      + 2.0*inv_sym6[2]*sym6[2]
      + inv_sym6[3]*sym6[3]
      + 2.0*inv_sym6[4]*sym6[4]
      + inv_sym6[5]*sym6[5]
    )

@jit(nopython=True)
def _norm2_sym6_jit(sym6, inv_sym6):
    """Compute A_ij A^ij"""
    return _trace_sym6_jit(sym6, inv_sym6)

@jit(nopython=True)
def _compute_hamiltonian_jit(R, gamma_sym6, K_sym6, Lambda):
    """JIT-compiled Hamiltonian constraint computation."""
    Nx, Ny, Nz = gamma_sym6.shape[:3]
    H = np.zeros((Nx, Ny, Nz))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                gamma_inv = _inv_sym6_jit(gamma_sym6[i,j,k])
                K_trace = _trace_sym6_jit(K_sym6[i,j,k], gamma_inv)
                K_sq = _norm2_sym6_jit(K_sym6[i,j,k], gamma_inv)
                H[i,j,k] = R[i,j,k] + K_trace**2 - K_sq - 2.0 * Lambda
    return H

@jit(nopython=True)
def _compute_momentum_jit(gamma_sym6, K_sym6, christoffels, dx, dy, dz):
    """JIT-compiled momentum constraint computation."""
    Nx, Ny, Nz = gamma_sym6.shape[:3]
    M = np.zeros((Nx, Ny, Nz, 3))
    S = np.zeros((Nx, Ny, Nz, 3, 3))

    # First, compute S^{ij} = K^{ij} - γ^{ij} K everywhere
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                gamma_inv = _inv_sym6_jit(gamma_sym6[i,j,k])
                gamma_inv_full = _sym6_to_mat33_jit(gamma_inv)
                K_trace = _trace_sym6_jit(K_sym6[i,j,k], gamma_inv)
                K_full = _sym6_to_mat33_jit(K_sym6[i,j,k])
                K_contravariant = gamma_inv_full @ K_full @ gamma_inv_full
                S[i,j,k] = K_contravariant - K_trace * gamma_inv_full

    # Now compute ∂_j S^{ij} using central differences
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for m in range(3):  # for each component m of M
                    div = 0.0
                    for n in range(3):  # sum over j = n
                        if n == 0:  # x direction
                            if i > 0 and i < Nx-1:
                                dS = (S[i+1,j,k,m,n] - S[i-1,j,k,m,n]) / (2 * dx)
                            else:
                                dS = 0.0
                        elif n == 1:  # y direction
                            if j > 0 and j < Ny-1:
                                dS = (S[i,j+1,k,m,n] - S[i,j-1,k,m,n]) / (2 * dy)
                            else:
                                dS = 0.0
                        elif n == 2:  # z direction
                            if k > 0 and k < Nz-1:
                                dS = (S[i,j,k+1,m,n] - S[i,j,k-1,m,n]) / (2 * dz)
                            else:
                                dS = 0.0
                        div += dS
                    M[i,j,k,m] = div

    return M

@jit(nopython=True)
def _discrete_L2_norm_jit(field, dx, dy, dz):
    """Compute discrete L2 norm: sqrt( sum(field^2) * dV )"""
    dV = dx * dy * dz
    sum_sq = 0.0
    for i in range(field.size):
        sum_sq += field.flat[i]**2
    return np.sqrt(sum_sq * dV)

def compute_hamiltonian_compiled(R, gamma_sym6, K_sym6, Lambda):
    """Compiled wrapper for Hamiltonian computation."""
    return _compute_hamiltonian_jit(R, gamma_sym6, K_sym6, Lambda)

def compute_momentum_compiled(gamma_sym6, K_sym6, christoffels, dx, dy, dz):
    """Compiled wrapper for momentum computation."""
    return _compute_momentum_jit(gamma_sym6, K_sym6, christoffels, dx, dy, dz)

def discrete_L2_norm_compiled(field, dx, dy, dz):
    """Compiled wrapper for L2 norm."""
    return _discrete_L2_norm_jit(field, dx, dy, dz)

# NSC runtime for constraints
def load_constraints_ir() -> Dict[str, Any]:
    """Mock IR for constraint computations."""
    return {
        "schema": "nsc_ir_v0.1",
        "op": {
            "expr": {
                "kind": "constraints_bundle",
            }
        }
    }

def make_constraints_callable(ir: Dict[str, Any]) -> Callable[..., Dict[str, np.ndarray]]:
    """Create callable for constraint computations."""

    def constraints_ops(R, gamma_sym6, K_sym6, Lambda, christoffels, dx, dy, dz):
        H = compute_hamiltonian_compiled(R, gamma_sym6, K_sym6, Lambda)
        M = compute_momentum_compiled(gamma_sym6, K_sym6, christoffels, dx, dy, dz)
        return {
            'H': H,
            'M': M
        }

    return constraints_ops