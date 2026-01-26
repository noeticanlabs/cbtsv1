#!/usr/bin/env python3
"""
NSC-compiled geometry computations for GR solver.
Provides JIT-compiled versions of geometry operations for performance.
"""

import numpy as np
from typing import Callable, Optional, Dict, Any
try:

    from numba import jit, prange
except ImportError:
    jit = lambda f=None, **kwargs: f if f else (lambda g: g)
    prange = range

# NSC Data Layout Constants (Source of Truth for sym6 packing)
IDX_XX = 0
IDX_XY = 1
IDX_XZ = 2
IDX_YY = 3
IDX_YZ = 4
IDX_ZZ = 5

def _fd_derivative_periodic(f, h, axis):
    """
    Computes the second-order centered finite difference with periodic boundary conditions.
    """
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2.0 * h)

# Geometry computation functions (compiled)
@jit(nopython=True)
def _sym6_to_mat33_jit(sym6):
    mat = np.zeros((3, 3), dtype=sym6.dtype)
    mat[0, 0] = sym6[IDX_XX]
    mat[0, 1] = sym6[IDX_XY]
    mat[0, 2] = sym6[IDX_XZ]
    mat[1, 0] = sym6[IDX_XY]
    mat[1, 1] = sym6[IDX_YY]
    mat[1, 2] = sym6[IDX_YZ]
    mat[2, 0] = sym6[IDX_XZ]
    mat[2, 1] = sym6[IDX_YZ]
    mat[2, 2] = sym6[IDX_ZZ]
    return mat

@jit(nopython=True)
def _inv_sym6_jit(sym6):
    xx, xy, xz, yy, yz, zz = sym6
    det = xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz) + xz * (xy * yz - yy * xz)
    
    # Guard against small determinant to avoid division by zero
    if abs(det) < 1e-15:
        det = 1e-15 if det >= 0 else -1e-15
        
    inv = np.empty(6, dtype=sym6.dtype)
    inv[0] = (yy*zz - yz*yz) / det
    inv[1] = -(xy*zz - xz*yz) / det
    inv[2] = (xy*yz - xz*yy) / det
    inv[3] = (xx*zz - xz*xz) / det
    inv[4] = -(xx*yz - xy*xz) / det
    inv[5] = (xx*yy - xy*xy) / det
    return inv

@jit(nopython=True, parallel=True, fastmath=True)
def _compute_christoffels_jit(Nx, Ny, Nz, gamma_sym6, dgamma_dx_sym6, dgamma_dy_sym6, dgamma_dz_sym6):
    """JIT-compiled computation of Christoffel symbols."""
    christoffels = np.zeros((Nx, Ny, Nz, 3, 3, 3))
    Gamma = np.zeros((Nx, Ny, Nz, 3))

    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                gamma_inv_full = _sym6_to_mat33_jit(_inv_sym6_jit(gamma_sym6[i, j, k]))

                dgamma_dx_mat = _sym6_to_mat33_jit(dgamma_dx_sym6[i, j, k])
                dgamma_dy_mat = _sym6_to_mat33_jit(dgamma_dy_sym6[i, j, k])
                dgamma_dz_mat = _sym6_to_mat33_jit(dgamma_dz_sym6[i, j, k])

                dgamma = np.empty((3, 3, 3), dtype=gamma_sym6.dtype)
                dgamma[0] = dgamma_dx_mat
                dgamma[1] = dgamma_dy_mat
                dgamma[2] = dgamma_dz_mat

                # Direct implementation of Gamma^i_{jk} = 0.5 * g^{il} * (d_j g_{kl} + d_k g_{jl} - d_l g_{jk})
                for i_idx in range(3):
                    for j_idx in range(3):
                        for k_idx in range(3):
                            sum_val = 0.0
                            for l_idx in range(3):
                                term = (dgamma[j_idx, l_idx, k_idx] +
                                        dgamma[k_idx, j_idx, l_idx] -
                                        dgamma[l_idx, j_idx, k_idx])
                                sum_val += gamma_inv_full[i_idx, l_idx] * term
                            christoffels[i, j, k, i_idx, j_idx, k_idx] = 0.5 * sum_val
                
                # Recompute contracted Gamma: Gamma^i = g^{jk} Gamma^i_{jk}
                for i_idx in range(3):
                    sum_val = 0.0
                    for j_idx in range(3):
                        for k_idx in range(3):
                            sum_val += gamma_inv_full[j_idx, k_idx] * christoffels[i, j, k, i_idx, j_idx, k_idx]
                    Gamma[i, j, k, i_idx] = sum_val

    return christoffels, Gamma

@jit(nopython=True, parallel=True, fastmath=True)
def _compute_ricci_jit(Nx, Ny, Nz, christoffels, d_christ_dx, d_christ_dy, d_christ_dz):
    """JIT-compiled Ricci tensor computation."""
    ricci = np.zeros((Nx, Ny, Nz, 3, 3))
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                Gamma = christoffels[i, j, k]
                dGamma_dx = d_christ_dx[i, j, k]
                dGamma_dy = d_christ_dy[i, j, k]
                dGamma_dz = d_christ_dz[i, j, k]
                
                # Precompute contracted Gamma for Term 3: C[lam] = Gamma^sigma_{lam, sigma}
                contracted_Gamma = np.zeros(3)
                for lam in range(3):
                    sum_sigma = 0.0
                    for sigma in range(3):
                        sum_sigma += Gamma[sigma, lam, sigma]
                    contracted_Gamma[lam] = sum_sigma

                for mu in range(3):
                    for nu in range(3):
                        # Term 1: partial_lambda Gamma^lambda_{mu, nu}
                        t1 = dGamma_dx[0, mu, nu] + dGamma_dy[1, mu, nu] + dGamma_dz[2, mu, nu]
                        
                        # Term 2: - partial_nu Gamma^lambda_{mu, lambda}
                        t2 = 0.0
                        if nu == 0:
                            dG_nu = dGamma_dx
                        elif nu == 1:
                            dG_nu = dGamma_dy
                        else:
                            dG_nu = dGamma_dz
                        
                        for lam in range(3):
                            t2 -= dG_nu[lam, mu, lam]
                        
                        # Term 3: Gamma^lambda_{mu, nu} * Gamma^sigma_{lambda, sigma}
                        t3 = 0.0
                        for lam in range(3):
                            t3 += Gamma[lam, mu, nu] * contracted_Gamma[lam]
                        
                        # Term 4: - Gamma^lambda_{mu, sigma} * Gamma^sigma_{nu, lambda}
                        t4 = 0.0
                        for lam in range(3):
                            for sigma in range(3):
                                t4 -= Gamma[lam, mu, sigma] * Gamma[sigma, nu, lam]
                        
                        ricci[i, j, k, mu, nu] = t1 + t2 + t3 + t4
    return ricci

@jit(nopython=True, parallel=True, fastmath=True)
def _second_covariant_derivative_scalar_jit(hess_scalar, christoffels, grad_scalar):
    """JIT-compiled second covariant derivative of scalar."""
    DD_scalar = np.zeros_like(hess_scalar)
    for i in range(hess_scalar.shape[0]):
        for j in range(hess_scalar.shape[1]):
            for k in range(hess_scalar.shape[2]):
                for l in range(3):
                    for m in range(3):
                        for n in range(3):
                            DD_scalar[i,j,k,l,m] = hess_scalar[i,j,k,l,m] - christoffels[i,j,k,n,l,m] * grad_scalar[i,j,k,n]
    return DD_scalar

@jit(nopython=True, parallel=True, fastmath=True)
def _lie_derivative_gamma_jit(gamma_full, beta, dgamma_d, grad_beta):
    """JIT-compiled Lie derivative of gamma."""
    Nx, Ny, Nz = gamma_full.shape[:3]
    lie_gamma_full = np.zeros_like(gamma_full)
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for l in range(3):
                    for m in range(3):
                        for n in range(3):
                            lie_gamma_full[i,j,k,l,m] += beta[i,j,k,n] * dgamma_d[n,i,j,k,l,m]
                            lie_gamma_full[i,j,k,l,m] += gamma_full[i,j,k,l,n] * grad_beta[i,j,k,m,n] + gamma_full[i,j,k,m,n] * grad_beta[i,j,k,l,n]

    lie_gamma_sym6 = np.zeros((Nx, Ny, Nz, 6), dtype=gamma_full.dtype)
    for i in prange(Nx):
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

@jit(nopython=True, parallel=True, fastmath=True)
def _lie_derivative_K_jit(K_full, beta, dK_d, grad_beta):
    """JIT-compiled Lie derivative of K."""
    Nx, Ny, Nz = K_full.shape[:3]
    lie_K_full = np.zeros_like(K_full)
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for l in range(3):
                    for m in range(3):
                        for n in range(3):
                            lie_K_full[i,j,k,l,m] += beta[i,j,k,n] * dK_d[n,i,j,k,l,m]
                            lie_K_full[i,j,k,l,m] += K_full[i,j,k,l,n] * grad_beta[i,j,k,m,n] + K_full[i,j,k,m,n] * grad_beta[i,j,k,l,n]

    lie_K_sym6 = np.zeros((Nx, Ny, Nz, 6), dtype=K_full.dtype)
    for i in prange(Nx):
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

def compute_christoffels_compiled(gamma_sym6, dgamma_dx, dgamma_dy, dgamma_dz):
    """Compiled wrapper for Christoffel computation."""
    Nx, Ny, Nz = gamma_sym6.shape[:3]
    return _compute_christoffels_jit(Nx, Ny, Nz, gamma_sym6, dgamma_dx, dgamma_dy, dgamma_dz)

def compute_ricci_compiled(christoffels, d_christ_dx, d_christ_dy, d_christ_dz):
    """Compiled wrapper for Ricci computation."""
    Nx, Ny, Nz = christoffels.shape[:3]
    return _compute_ricci_jit(Nx, Ny, Nz, christoffels, d_christ_dx, d_christ_dy, d_christ_dz)

def second_covariant_derivative_scalar_compiled(scalar, christoffels, dx, dy, dz):
    """Compiled wrapper for second covariant derivative."""
    grad_scalar = np.zeros(scalar.shape + (3,), dtype=scalar.dtype)
    grad_scalar[..., 0] = _fd_derivative_periodic(scalar, dx, axis=0)
    grad_scalar[..., 1] = _fd_derivative_periodic(scalar, dy, axis=1)
    grad_scalar[..., 2] = _fd_derivative_periodic(scalar, dz, axis=2)

    hess_scalar = np.zeros(scalar.shape + (3, 3), dtype=scalar.dtype)
    for i in range(3):
        hess_scalar[..., i, 0] = _fd_derivative_periodic(grad_scalar[..., i], dx, axis=0)
        hess_scalar[..., i, 1] = _fd_derivative_periodic(grad_scalar[..., i], dy, axis=1)
        hess_scalar[..., i, 2] = _fd_derivative_periodic(grad_scalar[..., i], dz, axis=2)

    return _second_covariant_derivative_scalar_jit(hess_scalar, christoffels, grad_scalar)

def lie_derivative_gamma_compiled(gamma_sym6, beta, dx, dy, dz):
    """Compiled wrapper for Lie derivative of gamma."""
    shape = gamma_sym6.shape[:-1]
    gamma_full = np.zeros(shape + (3, 3), dtype=gamma_sym6.dtype)
    gamma_full[..., 0, 0] = gamma_sym6[..., 0]; gamma_full[..., 0, 1] = gamma_sym6[..., 1]; gamma_full[..., 0, 2] = gamma_sym6[..., 2]
    gamma_full[..., 1, 0] = gamma_sym6[..., 1]; gamma_full[..., 1, 1] = gamma_sym6[..., 3]; gamma_full[..., 1, 2] = gamma_sym6[..., 4]
    gamma_full[..., 2, 0] = gamma_sym6[..., 2]; gamma_full[..., 2, 1] = gamma_sym6[..., 4]; gamma_full[..., 2, 2] = gamma_sym6[..., 5]

    dgamma_dx = _fd_derivative_periodic(gamma_full, dx, axis=0)
    dgamma_dy = _fd_derivative_periodic(gamma_full, dy, axis=1)
    dgamma_dz = _fd_derivative_periodic(gamma_full, dz, axis=2)
    dgamma_d = np.stack([dgamma_dx, dgamma_dy, dgamma_dz], axis=0)

    grad_beta = np.zeros(beta.shape + (3,), dtype=beta.dtype)
    for k in range(3):
        grad_beta[..., k, 0] = _fd_derivative_periodic(beta[..., k], dx, axis=0)
        grad_beta[..., k, 1] = _fd_derivative_periodic(beta[..., k], dy, axis=1)
        grad_beta[..., k, 2] = _fd_derivative_periodic(beta[..., k], dz, axis=2)

    return _lie_derivative_gamma_jit(gamma_full, beta, dgamma_d, grad_beta)

def lie_derivative_K_compiled(K_sym6, beta, dx, dy, dz):
    """Compiled wrapper for Lie derivative of K."""
    shape = K_sym6.shape[:-1]
    K_full = np.zeros(shape + (3, 3), dtype=K_sym6.dtype)
    K_full[..., 0, 0] = K_sym6[..., 0]; K_full[..., 0, 1] = K_sym6[..., 1]; K_full[..., 0, 2] = K_sym6[..., 2]
    K_full[..., 1, 0] = K_sym6[..., 1]; K_full[..., 1, 1] = K_sym6[..., 3]; K_full[..., 1, 2] = K_sym6[..., 4]
    K_full[..., 2, 0] = K_sym6[..., 2]; K_full[..., 2, 1] = K_sym6[..., 4]; K_full[..., 2, 2] = K_sym6[..., 5]

    dK_dx = _fd_derivative_periodic(K_full, dx, axis=0)
    dK_dy = _fd_derivative_periodic(K_full, dy, axis=1)
    dK_dz = _fd_derivative_periodic(K_full, dz, axis=2)
    dK_d = np.stack([dK_dx, dK_dy, dK_dz], axis=0)

    grad_beta = np.zeros(beta.shape + (3,), dtype=beta.dtype)
    for k in range(3):
        grad_beta[..., k, 0] = _fd_derivative_periodic(beta[..., k], dx, axis=0)
        grad_beta[..., k, 1] = _fd_derivative_periodic(beta[..., k], dy, axis=1)
        grad_beta[..., k, 2] = _fd_derivative_periodic(beta[..., k], dz, axis=2)

    return _lie_derivative_K_jit(K_full, beta, dK_d, grad_beta)

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
        dgamma_dx = _fd_derivative_periodic(gamma_sym6, dx, axis=0)
        dgamma_dy = _fd_derivative_periodic(gamma_sym6, dy, axis=1)
        dgamma_dz = _fd_derivative_periodic(gamma_sym6, dz, axis=2)
        christoffels, Gamma = compute_christoffels_compiled(gamma_sym6, dgamma_dx, dgamma_dy, dgamma_dz)
        
        d_christ_dx = _fd_derivative_periodic(christoffels, dx, axis=0)
        d_christ_dy = _fd_derivative_periodic(christoffels, dy, axis=1)
        d_christ_dz = _fd_derivative_periodic(christoffels, dz, axis=2)
        ricci = compute_ricci_compiled(christoffels, d_christ_dx, d_christ_dy, d_christ_dz)
        
        return {
            'christoffels': christoffels,
            'Gamma': Gamma,
            'ricci': ricci
        }

    return geometry_ops
