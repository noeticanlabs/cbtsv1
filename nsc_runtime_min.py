#!/usr/bin/env python3
import json
import numpy as np
from typing import Callable, Optional, Dict, Any
try:
    from numba import jit
except ImportError:
    jit = lambda f=None, **kwargs: f if f else (lambda g: g)

from gr_geometry_nsc import (
    _sym6_to_mat33_jit,
    _inv_sym6_jit,
    _compute_christoffels_jit,
    _compute_ricci_jit,
    _second_covariant_derivative_scalar_jit,
    _lie_derivative_gamma_jit,
    _lie_derivative_K_jit,
    compute_christoffels_compiled,
    compute_ricci_compiled,
    second_covariant_derivative_scalar_compiled,
    lie_derivative_gamma_compiled,
    lie_derivative_K_compiled
)
from gr_constraints_nsc import compute_hamiltonian_compiled, compute_momentum_compiled
from src.hadamard.vm import HadamardVM
from src.hadamard.compiler import HadamardCompiler

class NSCRuntimeError(RuntimeError):
    pass

@jit(nopython=True)
def _compute_gamma_tilde_rhs_jit(Nx, Ny, Nz, alpha, beta, phi, Gamma_tilde, A_sym6, gamma_tilde_sym6, dx, dy, dz, K_trace_scratch):
    """JIT-compiled computation of Gamma_tilde RHS."""
    # Precompute gamma_tilde_inv
    gamma_tilde_inv_full = np.zeros((Nx, Ny, Nz, 3, 3))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                gamma_tilde_inv_full[i,j,k] = _sym6_to_mat33_jit(_inv_sym6_jit(gamma_tilde_sym6[i,j,k]))

    # Precompute A_tilde_uu (contravariant)
    A_tilde_uu = np.zeros((Nx, Ny, Nz, 3, 3))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                A_full = _sym6_to_mat33_jit(A_sym6[i,j,k])
                A_tilde_uu[i,j,k] = gamma_tilde_inv_full[i,j,k] @ A_full @ gamma_tilde_inv_full[i,j,k]

    # Precompute Christoffel tildeGamma^i_{jk} - use the JIT christoffels
    christoffel_tilde_udd, _ = _compute_christoffels_jit(Nx, Ny, Nz, gamma_tilde_sym6, dx, dy, dz)

    # Gradients
    dalpha_x = np.gradient(alpha, dx, axis=0)
    dalpha_y = np.gradient(alpha, dy, axis=1)
    dalpha_z = np.gradient(alpha, dz, axis=2)
    dphi_x = np.gradient(phi, dx, axis=0)
    dphi_y = np.gradient(phi, dy, axis=1)
    dphi_z = np.gradient(phi, dz, axis=2)

    # Shift gradients
    dbeta = np.zeros((Nx, Ny, Nz, 3, 3))
    for k in range(3):
        dbeta[..., k, 0] = np.gradient(beta[..., k], dx, axis=0)
        dbeta[..., k, 1] = np.gradient(beta[..., k], dy, axis=1)
        dbeta[..., k, 2] = np.gradient(beta[..., k], dz, axis=2)

    # Second derivatives for C
    lap_beta = np.zeros((Nx, Ny, Nz, 3))
    for i in range(3):
        fxx = np.gradient(np.gradient(beta[..., i], dx, axis=0), dx, axis=0)
        fyy = np.gradient(np.gradient(beta[..., i], dy, axis=1), dy, axis=1)
        fzz = np.gradient(np.gradient(beta[..., i], dz, axis=2), dz, axis=2)
        lap_beta[..., i] = fxx + fyy + fzz

    div_beta = np.gradient(beta[..., 0], dx, axis=0) + np.gradient(beta[..., 1], dy, axis=1) + np.gradient(beta[..., 2], dz, axis=2)

    d_div_beta_x = np.gradient(div_beta, dx, axis=0)
    d_div_beta_y = np.gradient(div_beta, dy, axis=1)
    d_div_beta_z = np.gradient(div_beta, dz, axis=2)

    # Gamma gradients
    dGamma = np.zeros((Nx, Ny, Nz, 3, 3))
    for i in range(3):
        dGamma[..., i, 0] = np.gradient(Gamma_tilde[..., i], dx, axis=0)
        dGamma[..., i, 1] = np.gradient(Gamma_tilde[..., i], dy, axis=1)
        dGamma[..., i, 2] = np.gradient(Gamma_tilde[..., i], dz, axis=2)

    rhs_Gamma_tilde = np.zeros((Nx, Ny, Nz, 3))

    # A: advection
    rhs_Gamma_tilde += np.einsum('...k,...ik->...i', beta, dGamma)

    # B: stretching
    Gamma_dot_grad_beta = np.einsum('...k,...ik->...i', Gamma_tilde, dbeta)
    rhs_Gamma_tilde += -Gamma_dot_grad_beta + (2.0/3.0) * Gamma_tilde * div_beta[..., np.newaxis]

    # C: shift second-derivatives (approximated)
    d_div_beta = np.array([d_div_beta_x, d_div_beta_y, d_div_beta_z])
    d_div_beta = d_div_beta.transpose(1,2,3,0)
    rhs_Gamma_tilde += lap_beta + (1.0/3.0) * np.einsum('...ij,...j->...i', gamma_tilde_inv_full, d_div_beta)

    # D: lapse/curvature
    # -2 A^{ij} d_j alpha
    dalpha = np.array([dalpha_x, dalpha_y, dalpha_z]).transpose(1,2,3,0)
    rhs_Gamma_tilde += -2.0 * np.einsum('...ij,...j->...i', A_tilde_uu, dalpha)

    # 2 alpha (Gamma^i_{jk} A^{jk} + 6 A^{ij} d_j phi - (2/3) gamma^{ij} d_j K)
    GammaA = np.einsum('...ijk,...jk->...i', christoffel_tilde_udd, A_tilde_uu)

    dphi = np.array([dphi_x, dphi_y, dphi_z]).transpose(1,2,3,0)
    A_dphi = np.einsum('...ij,...j->...i', A_tilde_uu, dphi)

    dK_x = np.gradient(K_trace_scratch, dx, axis=0)
    dK_y = np.gradient(K_trace_scratch, dy, axis=1)
    dK_z = np.gradient(K_trace_scratch, dz, axis=2)
    dK = np.array([dK_x, dK_y, dK_z]).transpose(1,2,3,0)
    gamma_dK = np.einsum('...ij,...j->...i', gamma_tilde_inv_full, dK)

    rhs_Gamma_tilde += 2.0 * alpha[..., np.newaxis] * (GammaA + 6.0 * A_dphi - (2.0/3.0) * gamma_dK)

    return rhs_Gamma_tilde

def load_nscir(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        ir = json.load(f)
    if ir.get("schema") != "nsc_ir_v0.1":
        raise NSCRuntimeError("Bad schema")
    return ir

def make_rhs_callable(ir_path: str) -> Callable[..., Dict[str, np.ndarray]]:
    ir = load_nscir(ir_path)
    op = ir["op"]
    if op["expr"]["kind"] != "gr_rhs_bundle":
        raise NSCRuntimeError("Unsupported expr kind")
    if op["bc"] != "periodic":
        raise NSCRuntimeError("Only periodic bc supported by this mini runtime")
    dim = int(op["dim"])
    params = op["params"]

    # Compile IR to Hadamard bytecode
    compiler = HadamardCompiler()
    bytecode = compiler.compile_from_ir(ir_path)

    def rhs(fields: Dict[str, Any], lambda_val: Optional[float]=None, sources_enabled: Optional[bool]=None) -> Dict[str, np.ndarray]:
        # Use Hadamard VM to compute RHS
        vm = HadamardVM(fields)
        vm.execute(bytecode)
        return vm.rhs

    rhs._bytecode = bytecode  # Store for inspection

    # Helper functions from gr_core_fields and gr_geometry
    def sym6_to_mat33(sym6):
        """Convert symmetric 3x3 tensor stored as sym6 into full 3x3 matrix."""
        shape = sym6.shape[:-1]
        mat = np.zeros(shape + (3, 3), dtype=sym6.dtype)
        mat[..., 0, 0] = sym6[..., 0]
        mat[..., 0, 1] = sym6[..., 1]
        mat[..., 0, 2] = sym6[..., 2]
        mat[..., 1, 0] = sym6[..., 1]
        mat[..., 1, 1] = sym6[..., 3]
        mat[..., 1, 2] = sym6[..., 4]
        mat[..., 2, 0] = sym6[..., 2]
        mat[..., 2, 1] = sym6[..., 4]
        mat[..., 2, 2] = sym6[..., 5]
        return mat

    def mat33_to_sym6(mat):
        """Convert full 3x3 matrix to sym6 storage."""
        sym6 = np.empty(mat.shape[:-2] + (6,), dtype=mat.dtype)
        sym6[..., 0] = mat[..., 0, 0]
        sym6[..., 1] = mat[..., 0, 1]
        sym6[..., 2] = mat[..., 0, 2]
        sym6[..., 3] = mat[..., 1, 1]
        sym6[..., 4] = mat[..., 1, 2]
        sym6[..., 5] = mat[..., 2, 2]
        return sym6

    def inv_sym6(sym6):
        """Inverse of symmetric 3x3 tensor in sym6 form."""
        xx, xy, xz, yy, yz, zz = np.moveaxis(sym6, -1, 0)
        det = xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz) + xz * (xy * yz - yy * xz)
        inv = np.empty_like(sym6)
        inv[..., 0] = (yy*zz - yz*yz) / det
        inv[..., 1] = -(xy*zz - xz*yz) / det
        inv[..., 2] = (xy*yz - xz*yy) / det
        inv[..., 3] = (xx*zz - xz*xz) / det
        inv[..., 4] = -(xx*yz - xy*xz) / det
        inv[..., 5] = (xx*yy - xy*xy) / det
        return inv

    def trace_sym6(sym6, inv_sym6):
        """Compute trace: gamma^{ij} A_ij"""
        return (
            inv_sym6[..., 0]*sym6[..., 0]
          + 2.0*inv_sym6[..., 1]*sym6[..., 1]
          + 2.0*inv_sym6[..., 2]*sym6[..., 2]
          + inv_sym6[..., 3]*sym6[..., 3]
          + 2.0*inv_sym6[..., 4]*sym6[..., 4]
          + inv_sym6[..., 5]*sym6[..., 5]
        )

    rhs._nsc_lane = op["lane"]
    rhs._nsc_effects = op["effects"]
    rhs._nsc_ir_hash = ir.get("ir_hash")
    return rhs