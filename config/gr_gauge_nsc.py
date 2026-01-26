#!/usr/bin/env python3
"""
NSC-compiled gauge evolution for GR solver.
Provides JIT-compiled versions of gauge evolution operations.
"""

import numpy as np
from typing import Callable, Optional, Dict, Any
try:
    from numba import jit
except ImportError:
    jit = lambda f=None, **kwargs: f if f else (lambda g: g)

# Helper functions
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
    return (
        inv_sym6[0]*sym6[0]
      + 2.0*inv_sym6[1]*sym6[1]
      + 2.0*inv_sym6[2]*sym6[2]
      + inv_sym6[3]*sym6[3]
      + 2.0*inv_sym6[4]*sym6[4]
      + inv_sym6[5]*sym6[5]
    )

@jit(nopython=True)
def _evolve_lapse_jit(alpha, gamma_sym6, K_sym6, dt):
    """JIT-compiled lapse evolution."""
    Nx, Ny, Nz = alpha.shape
    alpha_new = alpha.copy()
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                gamma_inv = _inv_sym6_jit(gamma_sym6[i,j,k])
                K_trace = _trace_sym6_jit(K_sym6[i,j,k], gamma_inv)
                alpha_new[i,j,k] += dt * (-2.0 * alpha[i,j,k] * K_trace)
    return alpha_new

@jit(nopython=True)
def _evolve_shift_jit(beta, alpha, Gamma, lambda_i, dt):
    """JIT-compiled shift evolution."""
    Nx, Ny, Nz = beta.shape[:3]
    beta_new = beta.copy()
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                B = Gamma[i,j,k] - lambda_i[i,j,k]
                for comp in range(3):
                    beta_new[i,j,k,comp] += dt * (0.75 * alpha[i,j,k] * B[comp])
    return beta_new

@jit(nopython=True)
def _compute_gradients_jit(alpha, beta, dx, dy, dz):
    """JIT-compiled gradient computation."""
    Nx, Ny, Nz = alpha.shape
    grad_alpha = np.zeros((Nx, Ny, Nz, 3))
    grad_alpha[..., 0] = np.gradient(alpha, dx, axis=0)
    grad_alpha[..., 1] = np.gradient(alpha, dy, axis=1)
    grad_alpha[..., 2] = np.gradient(alpha, dz, axis=2)

    grad_beta = np.zeros((Nx, Ny, Nz, 3, 3))
    for comp in range(3):
        grad_beta[..., comp, 0] = np.gradient(beta[..., comp], dx, axis=0)
        grad_beta[..., comp, 1] = np.gradient(beta[..., comp], dy, axis=1)
        grad_beta[..., comp, 2] = np.gradient(beta[..., comp], dz, axis=2)

    return grad_alpha, grad_beta

@jit(nopython=True)
def _compute_dt_gauge_jit(grad_alpha, grad_beta):
    """JIT-compiled dt_gauge computation."""
    # Magnitude of grad alpha
    grad_alpha_mag = np.sqrt(np.sum(grad_alpha**2, axis=-1))

    # Magnitude of each component's gradient
    grad_beta_mag = np.sqrt(np.sum(grad_beta**2, axis=-2))  # sum over deriv indices
    max_grad_beta = np.max(grad_beta_mag, axis=-1)  # max over components

    # Avoid division by zero
    dt_alpha = 1.0 / (np.sqrt(grad_alpha_mag) + 1e-15)
    dt_beta = 1.0 / (max_grad_beta + 1e-15)

    dt_gauge = np.minimum(dt_alpha, dt_beta).min()
    return dt_gauge

def evolve_lapse_compiled(alpha, gamma_sym6, K_sym6, dt):
    """Compiled wrapper for lapse evolution."""
    return _evolve_lapse_jit(alpha, gamma_sym6, K_sym6, dt)

def evolve_shift_compiled(beta, alpha, Gamma, lambda_i, dt):
    """Compiled wrapper for shift evolution."""
    return _evolve_shift_jit(beta, alpha, Gamma, lambda_i, dt)

def compute_gradients_compiled(alpha, beta, dx, dy, dz):
    """Compiled wrapper for gradient computation."""
    return _compute_gradients_jit(alpha, beta, dx, dy, dz)

def compute_dt_gauge_compiled(grad_alpha, grad_beta):
    """Compiled wrapper for dt_gauge computation."""
    return _compute_dt_gauge_jit(grad_alpha, grad_beta)

# NSC runtime for gauge
def load_gauge_ir() -> Dict[str, Any]:
    """Mock IR for gauge computations."""
    return {
        "schema": "nsc_ir_v0.1",
        "op": {
            "expr": {
                "kind": "gauge_bundle",
            }
        }
    }

def make_gauge_callable(ir: Dict[str, Any]) -> Callable[..., Dict[str, np.ndarray]]:
    """Create callable for gauge computations."""

    def gauge_ops(alpha, gamma_sym6, K_sym6, beta, Gamma, lambda_i, dx, dy, dz, dt, operation):
        if operation == 'evolve_lapse':
            return {'alpha': evolve_lapse_compiled(alpha, gamma_sym6, K_sym6, dt)}
        elif operation == 'evolve_shift':
            return {'beta': evolve_shift_compiled(beta, alpha, Gamma, lambda_i, dt)}
        elif operation == 'compute_gradients':
            grad_alpha, grad_beta = compute_gradients_compiled(alpha, beta, dx, dy, dz)
            return {'grad_alpha': grad_alpha, 'grad_beta': grad_beta}
        elif operation == 'compute_dt_gauge':
            grad_alpha, grad_beta = compute_gradients_compiled(alpha, beta, dx, dy, dz)
            dt_gauge = compute_dt_gauge_compiled(grad_alpha, grad_beta)
            return {'dt_gauge': dt_gauge}
        else:
            raise ValueError(f"Unknown operation: {operation}")

    return gauge_ops