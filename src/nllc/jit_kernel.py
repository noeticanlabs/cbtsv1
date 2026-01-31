"""
NSC JIT Kernel Binding

Binds NIR intrinsic calls to compiled Numba/NumPy kernels for GPU/HPC acceleration.

Author: CBTSV1 Team
Version: 1.0
"""

import numpy as np
from numba import cuda, jit, njit, prange
from typing import Callable, Dict, Any, Optional

from .nir import *

# Kernel registry: maps intrinsic name to implementation
KERNEL_REGISTRY: Dict[str, Callable] = {}


def register_kernel(name: str):
    """Decorator to register a JIT kernel."""
    def decorator(func: Callable):
        KERNEL_REGISTRY[name] = func
        return func
    return decorator


# =============================================================================
# Geometry Kernels
# =============================================================================

@register_kernel("inv_sym6")
@njit(cache=True, parallel=True)
def kernel_inv_sym6(gamma_sym6: np.ndarray) -> np.ndarray:
    """
    Compute inverse of symmetric 2-tensor from 6 components.
    
    gamma_sym6 shape: (..., 6) where components are [xx, yy, zz, xy, xz, yz]
    Returns: inverse tensor in same format
    """
    # Extract components
    gxx = gamma_sym6[..., 0]
    gyy = gamma_sym6[..., 1]
    gzz = gamma_sym6[..., 2]
    gxy = gamma_sym6[..., 3]
    gxz = gamma_sym6[..., 4]
    gyz = gamma_sym6[..., 5]
    
    # Build 3x3 matrix
    gamma = np.zeros((*gamma_sym6.shape[:-1], 3, 3))
    gamma[..., 0, 0] = gxx
    gamma[..., 1, 1] = gyy
    gamma[..., 2, 2] = gzz
    gamma[..., 0, 1] = gamma[..., 1, 0] = gxy
    gamma[..., 0, 2] = gamma[..., 2, 0] = gxz
    gamma[..., 1, 2] = gamma[..., 2, 1] = gyz
    
    # Compute inverse
    inv = np.zeros_like(gamma)
    for idx in np.ndindex(gamma.shape[:-2]):
        inv_idx = (*idx,)
        try:
            inv[inv_idx] = np.linalg.inv(gamma[inv_idx])
        except np.linalg.LinAlgError:
            inv[inv_idx] = np.eye(3)
    
    # Convert back to sym6 format
    result = np.zeros_like(gamma_sym6)
    result[..., 0] = inv[..., 0, 0]
    result[..., 1] = inv[..., 1, 1]
    result[..., 2] = inv[..., 2, 2]
    result[..., 3] = inv[..., 0, 1]
    result[..., 4] = inv[..., 0, 2]
    result[..., 5] = inv[..., 1, 2]
    
    return result


@register_kernel("trace_sym6")
@njit(cache=True, parallel=True)
def kernel_trace_sym6(gamma_sym6: np.ndarray) -> np.ndarray:
    """Compute trace of symmetric 2-tensor."""
    return gamma_sym6[..., 0] + gamma_sym6[..., 1] + gamma_sym6[..., 2]


@register_kernel("det_sym6")
@njit(cache=True, parallel=True)
def kernel_det_sym6(gamma_sym6: np.ndarray) -> np.ndarray:
    """Compute determinant of symmetric 2-tensor."""
    gxx = gamma_sym6[..., 0]
    gyy = gamma_sym6[..., 1]
    gzz = gamma_sym6[..., 2]
    gxy = gamma_sym6[..., 3]
    gxz = gamma_sym6[..., 4]
    gyz = gamma_sym6[..., 5]
    
    det = (gxx * gyy * gzz 
           + 2 * gxy * gxz * gyz 
           - gxx * gyz**2 
           - gyy * gxz**2 
           - gzz * gxy**2)
    return det


@register_kernel("norm2_sym6")
@njit(cache=True, parallel=True)
def kernel_norm2_sym6(tensor_sym6: np.ndarray, metric_sym6: np.ndarray) -> np.ndarray:
    """Compute L2 norm squared: γ^{ij} γ^{kl} T_{ik} T_{jl}."""
    # Compute inverse metric
    inv = kernel_inv_sym6(metric_sym6)
    
    # Compute norm: inv^{ij} * T_{ij} where T is the input tensor
    # For symmetric 2-tensor in 3D:
    norm = (inv[..., 0, 0] * tensor_sym6[..., 0] * tensor_sym6[..., 0]
            + inv[..., 1, 1] * tensor_sym6[..., 1] * tensor_sym6[..., 1]
            + inv[..., 2, 2] * tensor_sym6[..., 2] * tensor_sym6[..., 2]
            + 2 * inv[..., 0, 1] * tensor_sym6[..., 0] * tensor_sym6[..., 1]
            + 2 * inv[..., 0, 2] * tensor_sym6[..., 0] * tensor_sym6[..., 2]
            + 2 * inv[..., 1, 2] * tensor_sym6[..., 1] * tensor_sym6[..., 2])
    return norm


@register_kernel("sym6_to_mat33")
@njit(cache=True, parallel=True)
def kernel_sym6_to_mat33(sym6: np.ndarray) -> np.ndarray:
    """Convert 6-component symmetric tensor to 3x3 matrix."""
    mat = np.zeros((*sym6.shape[:-1], 3, 3))
    mat[..., 0, 0] = sym6[..., 0]
    mat[..., 1, 1] = sym6[..., 1]
    mat[..., 2, 2] = sym6[..., 2]
    mat[..., 0, 1] = mat[..., 1, 0] = sym6[..., 3]
    mat[..., 0, 2] = mat[..., 2, 0] = sym6[..., 4]
    mat[..., 1, 2] = mat[..., 2, 1] = sym6[..., 5]
    return mat


@register_kernel("mat33_to_sym6")
@njit(cache=True, parallel=True)
def kernel_mat33_to_sym6(mat33: np.ndarray) -> np.ndarray:
    """Convert 3x3 matrix to 6-component symmetric tensor."""
    sym6 = np.zeros((*mat33.shape[:-2], 6))
    sym6[..., 0] = mat33[..., 0, 0]
    sym6[..., 1] = mat33[..., 1, 1]
    sym6[..., 2] = mat33[..., 2, 2]
    sym6[..., 3] = mat33[..., 0, 1]
    sym6[..., 4] = mat33[..., 0, 2]
    sym6[..., 5] = mat33[..., 1, 2]
    return sym6


# =============================================================================
# Constraint Kernels
# =============================================================================

@register_kernel("christoffel_symbols")
@njit(cache=True, parallel=True)
def kernel_christoffel(gamma_sym6: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute Christoffel symbols using finite differences.
    
    Returns: Gamma^k_ij in flattened format [i,j,k] -> [k_idx]
    """
    # This is a simplified version - full implementation would use
    # inverse metric derivatives
    inv = kernel_inv_sym6(gamma_sym6)
    Gamma = np.zeros((*gamma_sym6.shape[:-1], 27))  # 3x3x3 = 27 components
    
    # Placeholder: return zeros
    return Gamma


@register_kernel("hamiltonian_constraint")
@njit(cache=True, parallel=True)
def kernel_hamiltonian_constraint(
    gamma_sym6: np.ndarray, 
    K_sym6: np.ndarray,
    alpha: np.ndarray,
    phi: np.ndarray
) -> np.ndarray:
    """
    Compute Hamiltonian constraint residual.
    
    H = R + K^2 - K_ij K^ij - 16πρ
    (simplified form for vacuum: H = R + K^2 - K_ij K^ij)
    """
    # Compute trace K
    K_trace = kernel_trace_sym6(K_sym6)
    
    # Compute K_ij K^ij
    K_squared = kernel_norm2_sym6(K_sym6, gamma_sym6)
    
    # Placeholder: return zero constraint (full implementation would
    # include Ricci scalar from Christoffel symbols)
    H = np.zeros(gamma_sym6.shape[:-1])
    
    return H


@register_kernel("momentum_constraint")
@njit(cache=True, parallel=True)
def kernel_momentum_constraint(
    gamma_sym6: np.ndarray,
    K_sym6: np.ndarray,
    beta: np.ndarray
) -> np.ndarray:
    """
    Compute momentum constraint residual.
    
    M^i = D_j (K^{ij} - γ^{ij} K)
    """
    # Placeholder: return zero constraint
    M = np.zeros((*gamma_sym6.shape[:-1], 3))
    return M


# =============================================================================
# Spectral Kernels
# =============================================================================

@register_kernel("fft_power_spectrum")
@njit(cache=True, parallel=True)
def kernel_fft_power_spectrum(field: np.ndarray) -> np.ndarray:
    """
    Compute FFT power spectrum for spectral analysis.
    
    Returns: Power in k-space bins
    """
    # Use numpy FFT (not parallelized in Numba)
    fft_result = np.fft.fftn(field.astype(np.complex128))
    power = np.abs(fft_result)**2
    
    # Return full spectrum
    return power


# =============================================================================
# JIT Kernel Binder
# =============================================================================

class JITKernelBinder:
    """
    Binds NIR CallInst intrinsics to JIT-compiled kernels.
    
    Usage:
        binder = JITKernelBinder()
        result = binder.bind_call(call_inst, args)
    """
    
    def __init__(self):
        self.registry = KERNEL_REGISTRY
    
    def bind_call(self, call: CallInst, args: list) -> np.ndarray:
        """
        Bind a NIR call instruction to a kernel.
        
        Args:
            call: CallInst from NIR
            args: List of numpy arrays (operands)
        
        Returns:
            Result array from kernel execution
        """
        func_name = call.func
        
        if func_name not in self.registry:
            raise ValueError(f"Unknown kernel: {func_name}")
        
        kernel = self.registry[func_name]
        
        # Call kernel with arguments
        result = kernel(*args)
        
        return result
    
    def bind_module(self, module: Module) -> Module:
        """
        Process a NIR module, replacing CallInst with kernel bindings.
        
        Returns modified module with kernel results.
        """
        # For now, this is a pass-through
        # Full implementation would inline kernel results
        return module


def bind_jit_kernel(func_name: str, kernel: Callable):
    """Register a kernel for JIT binding."""
    KERNEL_REGISTRY[func_name] = kernel


def get_kernel(func_name: str) -> Optional[Callable]:
    """Get a registered kernel."""
    return KERNEL_REGISTRY.get(func_name)
