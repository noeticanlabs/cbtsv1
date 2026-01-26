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
    """Compute A_ij A^ij = A_ij gamma^ik gamma^jl A_kl"""
    A_mat = _sym6_to_mat33_jit(sym6)
    ginv_mat = _sym6_to_mat33_jit(inv_sym6)
    
    # A_up_mat corresponds to A^{ij}
    A_up_mat = np.zeros((3, 3), dtype=sym6.dtype)
    for i in range(3):
        for j in range(3):
            val = 0.0
            for k in range(3):
                for l in range(3):
                    val += ginv_mat[i, k] * ginv_mat[j, l] * A_mat[k, l]
            A_up_mat[i, j] = val

    # Contract A_ij with A^{ij}
    norm_sq = 0.0
    for i in range(3):
        for j in range(3):
            norm_sq += A_mat[i, j] * A_up_mat[i, j]
            
    return norm_sq

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

def _compute_momentum_py(gamma_sym6, K_sym6, christoffels, grad_log_sqrt_det_gamma, dx, dy, dz):
    """Pure Python momentum constraint computation."""
    Nx, Ny, Nz = gamma_sym6.shape[:3]
    S_ij = np.zeros((Nx, Ny, Nz, 3, 3))

    # This is slow, but avoids Numba issues.
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                gamma_inv = _inv_sym6_jit(gamma_sym6[i,j,k])
                gamma_inv_full = _sym6_to_mat33_jit(gamma_inv)
                K_trace = _trace_sym6_jit(K_sym6[i,j,k], gamma_inv)
                K_full = _sym6_to_mat33_jit(K_sym6[i,j,k])
                K_contravariant = gamma_inv_full @ K_full @ gamma_inv_full
                S_ij[i,j,k] = K_contravariant - K_trace * gamma_inv_full

    partial_div = np.zeros((Nx, Ny, Nz, 3))
    for i in range(3):
        partial_div[..., i] = (_fd_derivative_periodic(S_ij[..., i, 0], dx, 0) +
                               _fd_derivative_periodic(S_ij[..., i, 1], dy, 1) +
                               _fd_derivative_periodic(S_ij[..., i, 2], dz, 2))

    gamma_term1 = np.zeros((Nx, Ny, Nz, 3))
    gamma_term2 = np.zeros((Nx, Ny, Nz, 3))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for i_comp in range(3):
                    for j_comp in range(3):
                        for k_comp in range(3):
                            gamma_term1[i,j,k,i_comp] += christoffels[i,j,k,i_comp,j_comp,k_comp] * S_ij[i,j,k,j_comp,k_comp]
                for i_comp in range(3):
                    for k_comp in range(3):
                        gamma_term2[i,j,k,i_comp] += grad_log_sqrt_det_gamma[i,j,k,k_comp] * S_ij[i,j,k,i_comp,k_comp]

    M = partial_div + gamma_term1 + gamma_term2
    return M

@jit(nopython=True)
def _discrete_L2_norm_jit(field, dx, dy, dz):
    """Compute discrete L2 norm: sqrt( sum(field^2) * dV )"""
    dV = dx * dy * dz
    sum_sq = 0.0
    for i in range(field.size):
        sum_sq += field.flat[i]**2
    return np.sqrt(sum_sq * dV)

def _fd_derivative_periodic(f, h, axis):
    """Computes the second-order centered finite difference with periodic boundary conditions."""
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2.0 * h)

def _det_sym6_array(sym6):
    """Compute determinant of sym6 array (Nx, Ny, Nz, 6)."""
    xx = sym6[..., 0]
    xy = sym6[..., 1]
    xz = sym6[..., 2]
    yy = sym6[..., 3]
    yz = sym6[..., 4]
    zz = sym6[..., 5]
    return xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz) + xz * (xy * yz - yy * xz)

def compute_hamiltonian_compiled(R, gamma_sym6, K_sym6, Lambda):
    """Compiled wrapper for Hamiltonian computation."""
    return _compute_hamiltonian_jit(R, gamma_sym6, K_sym6, Lambda)

def compute_momentum_compiled(gamma_sym6, K_sym6, christoffels, dx, dy, dz, det_sym6_func):
    """Wrapper for momentum computation. Falls back to Python due to Numba issue."""
    det_gamma = det_sym6_func(gamma_sym6)
    det_gamma = np.maximum(det_gamma, 1e-16)
    log_sqrt_det_gamma = 0.5 * np.log(det_gamma)
    
    g_x = _fd_derivative_periodic(log_sqrt_det_gamma, dx, axis=0)
    g_y = _fd_derivative_periodic(log_sqrt_det_gamma, dy, axis=1)
    g_z = _fd_derivative_periodic(log_sqrt_det_gamma, dz, axis=2)
    grad_log_sqrt_det_gamma = np.stack([g_x, g_y, g_z], axis=-1)
    
    return _compute_momentum_py(gamma_sym6, K_sym6, christoffels, grad_log_sqrt_det_gamma, dx, dy, dz)

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
        M = compute_momentum_compiled(gamma_sym6, K_sym6, christoffels, dx, dy, dz, _det_sym6_array)
        return {
            'H': H,
            'M': M
        }

    return constraints_ops
