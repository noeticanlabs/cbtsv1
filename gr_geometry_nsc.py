#!/usr/bin/env python3
"""
NSC-compiled geometry computations for GR solver.
Provides JIT-compiled versions of geometry operations for performance.
"""

import numpy as np
from typing import Callable, Optional, Dict, Any
try:
    from numba import jit
except ImportError:
    jit = lambda f=None, **kwargs: f if f else (lambda g: g)

# Geometry computation functions (compiled)
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
def _compute_christoffels_jit(Nx, Ny, Nz, gamma_sym6, dx, dy, dz):
    """JIT-compiled computation of Christoffel symbols."""
    gamma_full = np.zeros((Nx, Ny, Nz, 3, 3))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                gamma_full[i,j,k] = _sym6_to_mat33_jit(gamma_sym6[i,j,k])

    dgamma_dx = np.zeros((Nx, Ny, Nz, 3, 3))
    dgamma_dy = np.zeros((Nx, Ny, Nz, 3, 3))
    dgamma_dz = np.zeros((Nx, Ny, Nz, 3, 3))
    for i in range(3):
        for j in range(3):
            dgamma_dx[:, :, :, i, j] = np.gradient(gamma_full[:, :, :, i, j], dx, axis=0)
            dgamma_dy[:, :, :, i, j] = np.gradient(gamma_full[:, :, :, i, j], dy, axis=1)
            dgamma_dz[:, :, :, i, j] = np.gradient(gamma_full[:, :, :, i, j], dz, axis=2)

    gamma_inv_full = np.zeros((Nx, Ny, Nz, 3, 3))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                gamma_inv_full[i,j,k] = _sym6_to_mat33_jit(_inv_sym6_jit(gamma_sym6[i,j,k]))

    dgamma = np.zeros((Nx, Ny, Nz, 3, 3, 3))
    dgamma[..., 0, :, :] = dgamma_dx
    dgamma[..., 1, :, :] = dgamma_dy
    dgamma[..., 2, :, :] = dgamma_dz

    T = np.zeros((Nx, Ny, Nz, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for l in range(3):
                T[..., i, j, l] = dgamma[..., i, j, l] + dgamma[..., j, i, l] - dgamma[..., l, i, j]

    christoffels = np.zeros((Nx, Ny, Nz, 3, 3, 3))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                christoffels[i,j,k] = 0.5 * gamma_inv_full[i,j,k] @ T[i,j,k]

    Gamma = np.zeros((Nx, Ny, Nz, 3))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                Gamma[i,j,k] = np.sum(gamma_inv_full[i,j,k] * christoffels[i,j,k], axis=(0,1))

    return christoffels, Gamma

@jit(nopython=True)
def _compute_ricci_jit(Nx, Ny, Nz, gamma_sym6, christoffels, dx, dy, dz):
    """JIT-compiled Ricci tensor computation."""
    d_christ_dx = np.zeros((Nx, Ny, Nz, 3, 3, 3))
    d_christ_dy = np.zeros((Nx, Ny, Nz, 3, 3, 3))
    d_christ_dz = np.zeros((Nx, Ny, Nz, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                d_christ_dx[:, :, :, i, j, k] = np.gradient(christoffels[:, :, :, i, j, k], dx, axis=0)
                d_christ_dy[:, :, :, i, j, k] = np.gradient(christoffels[:, :, :, i, j, k], dy, axis=1)
                d_christ_dz[:, :, :, i, j, k] = np.gradient(christoffels[:, :, :, i, j, k], dz, axis=2)

    term1 = d_christ_dx.sum(axis=3) + d_christ_dy.sum(axis=3) + d_christ_dz.sum(axis=3)
    term2 = np.zeros((Nx, Ny, Nz, 3, 3))
    for j in range(3):
        d_christ = [d_christ_dx, d_christ_dy, d_christ_dz][j]
        for i in range(3):
            term2[..., i, j] = -d_christ[..., np.arange(3), i, np.arange(3)].sum(axis=-1)
    term3 = np.einsum('...kij,...lkl->...ij', christoffels, christoffels)
    term4 = np.einsum('...kil,...lkj->...ij', christoffels, christoffels)
    ricci = term1 + term2 + term3 - term4
    return ricci

@jit(nopython=True)
def _second_covariant_derivative_scalar_jit(scalar, christoffels, dx, dy, dz):
    """JIT-compiled second covariant derivative of scalar."""
    Nx, Ny, Nz = scalar.shape
    grad_scalar = np.zeros((Nx, Ny, Nz, 3))
    grad_scalar[..., 0] = np.gradient(scalar, dx, axis=0)
    grad_scalar[..., 1] = np.gradient(scalar, dy, axis=1)
    grad_scalar[..., 2] = np.gradient(scalar, dz, axis=2)
    hess_scalar = np.zeros((Nx, Ny, Nz, 3, 3))
    for i in range(3):
        hess_scalar[..., i, 0] = np.gradient(grad_scalar[..., i], dx, axis=0)
        hess_scalar[..., i, 1] = np.gradient(grad_scalar[..., i], dy, axis=1)
        hess_scalar[..., i, 2] = np.gradient(grad_scalar[..., i], dz, axis=2)
    DD_scalar = hess_scalar - np.einsum('...kij,...k->...ij', christoffels, grad_scalar)
    return DD_scalar

@jit(nopython=True)
def _lie_derivative_gamma_jit(gamma_sym6, beta, dx, dy, dz):
    """JIT-compiled Lie derivative of gamma."""
    Nx, Ny, Nz = gamma_sym6.shape[:3]
    gamma_full = np.zeros((Nx, Ny, Nz, 3, 3))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                gamma_full[i,j,k] = _sym6_to_mat33_jit(gamma_sym6[i,j,k])

    dgamma_dx = np.gradient(gamma_full, dx, axis=0)
    dgamma_dy = np.gradient(gamma_full, dy, axis=1)
    dgamma_dz = np.gradient(gamma_full, dz, axis=2)
    dgamma_d = np.stack([dgamma_dx, dgamma_dy, dgamma_dz], axis=0)
    lie_term1 = np.einsum('...k,k...ij->...ij', beta, dgamma_d)
    grad_beta = np.zeros((Nx, Ny, Nz, 3, 3))
    for k in range(3):
        grad_beta[..., k, 0] = np.gradient(beta[..., k], dx, axis=0)
        grad_beta[..., k, 1] = np.gradient(beta[..., k], dy, axis=1)
        grad_beta[..., k, 2] = np.gradient(beta[..., k], dz, axis=2)
    lie_term2 = np.einsum('...kj,...ki->...ij', gamma_full, grad_beta) + np.einsum('...ik,...kj->...ij', gamma_full, grad_beta)
    lie_gamma_full = lie_term1 + lie_term2
    lie_gamma_sym6 = np.zeros_like(gamma_sym6)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                mat = lie_gamma_full[i,j,k]
                lie_gamma_sym6[i,j,k,0] = mat[0,0]
                lie_gamma_sym6[i,j,k,1] = mat[0,1]
                lie_gamma_sym6[i,j,k,2] = mat[0,2]
                lie_gamma_sym6[i,j,k,3] = mat[1,1]
                lie_gamma_sym6[i,j,k,4] = mat[1,2]
                lie_gamma_sym6[i,j,k,5] = mat[2,2]
    return lie_gamma_sym6

@jit(nopython=True)
def _lie_derivative_K_jit(K_sym6, beta, dx, dy, dz):
    """JIT-compiled Lie derivative of K."""
    Nx, Ny, Nz = K_sym6.shape[:3]
    K_full = np.zeros((Nx, Ny, Nz, 3, 3))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                K_full[i,j,k] = _sym6_to_mat33_jit(K_sym6[i,j,k])

    dK_dx = np.gradient(K_full, dx, axis=0)
    dK_dy = np.gradient(K_full, dy, axis=1)
    dK_dz = np.gradient(K_full, dz, axis=2)
    dK_d = np.stack([dK_dx, dK_dy, dK_dz], axis=0)
    lie_term1 = np.einsum('...k,k...ij->...ij', beta, dK_d)
    grad_beta = np.zeros((Nx, Ny, Nz, 3, 3))
    for k in range(3):
        grad_beta[..., k, 0] = np.gradient(beta[..., k], dx, axis=0)
        grad_beta[..., k, 1] = np.gradient(beta[..., k], dy, axis=1)
        grad_beta[..., k, 2] = np.gradient(beta[..., k], dz, axis=2)
    lie_term2 = np.einsum('...kj,...ki->...ij', K_full, grad_beta) + np.einsum('...ik,...kj->...ij', K_full, grad_beta)
    lie_K_full = lie_term1 + lie_term2
    lie_K_sym6 = np.zeros_like(K_sym6)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                mat = lie_K_full[i,j,k]
                lie_K_sym6[i,j,k,0] = mat[0,0]
                lie_K_sym6[i,j,k,1] = mat[0,1]
                lie_K_sym6[i,j,k,2] = mat[0,2]
                lie_K_sym6[i,j,k,3] = mat[1,1]
                lie_K_sym6[i,j,k,4] = mat[1,2]
                lie_K_sym6[i,j,k,5] = mat[2,2]
    return lie_K_sym6

def compute_christoffels_compiled(gamma_sym6, dx, dy, dz):
    """Compiled wrapper for Christoffel computation."""
    Nx, Ny, Nz = gamma_sym6.shape[:3]
    return _compute_christoffels_jit(Nx, Ny, Nz, gamma_sym6, dx, dy, dz)

def compute_ricci_compiled(gamma_sym6, christoffels, dx, dy, dz):
    """Compiled wrapper for Ricci computation."""
    Nx, Ny, Nz = gamma_sym6.shape[:3]
    return _compute_ricci_jit(Nx, Ny, Nz, gamma_sym6, christoffels, dx, dy, dz)

def second_covariant_derivative_scalar_compiled(scalar, christoffels, dx, dy, dz):
    """Compiled wrapper for second covariant derivative."""
    return _second_covariant_derivative_scalar_jit(scalar, christoffels, dx, dy, dz)

def lie_derivative_gamma_compiled(gamma_sym6, beta, dx, dy, dz):
    """Compiled wrapper for Lie derivative of gamma."""
    return _lie_derivative_gamma_jit(gamma_sym6, beta, dx, dy, dz)

def lie_derivative_K_compiled(K_sym6, beta, dx, dy, dz):
    """Compiled wrapper for Lie derivative of K."""
    return _lie_derivative_K_jit(K_sym6, beta, dx, dy, dz)

@jit(nopython=True)
def _compute_ricci_scalar_jit(ricci_tensor, gamma_sym6):
    """JIT-compiled Ricci scalar computation."""
    Nx, Ny, Nz = gamma_sym6.shape[:3]
    R_scalar = np.zeros((Nx, Ny, Nz))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                gamma_inv = _inv_sym6_jit(gamma_sym6[i,j,k])
                gamma_inv_mat = _sym6_to_mat33_jit(gamma_inv)
                R_scalar[i,j,k] = np.sum(gamma_inv_mat * ricci_tensor[i,j,k])
    return R_scalar

def compute_ricci_scalar_compiled(ricci_tensor, gamma_sym6):
    """Compiled wrapper for Ricci scalar computation."""
    return _compute_ricci_scalar_jit(ricci_tensor, gamma_sym6)

# NSC runtime for geometry
def load_geometry_ir() -> Dict[str, Any]:
    """Mock IR for geometry computations."""
    return {
        "schema": "nsc_ir_v0.1",
        "op": {
            "expr": {
                "kind": "geometry_bundle",
            }
        }
    }

def make_geometry_callable(ir: Dict[str, Any]) -> Callable[..., Dict[str, np.ndarray]]:
    """Create callable for geometry computations."""

    def geometry_ops(gamma_sym6, dx, dy, dz):
        christoffels, Gamma = compute_christoffels_compiled(gamma_sym6, dx, dy, dz)
        ricci = compute_ricci_compiled(gamma_sym6, christoffels, dx, dy, dz)
        return {
            'christoffels': christoffels,
            'Gamma': Gamma,
            'ricci': ricci
        }

    return geometry_ops