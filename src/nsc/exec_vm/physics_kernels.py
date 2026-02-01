"""
Physics Kernels (Numba-accelerated) for NSC-M3L Virtual Machine

Implements gradient, divergence, curl, and Laplacian operations.
"""

import numpy as np


# =============================================================================
# Numba Import with Fallback
# =============================================================================

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback: define identity decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range


# =============================================================================
# Physics Kernels
# =============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def _compute_gradient_jit(field: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
        """Compute gradient with central differences."""
        nx, ny, nz = field.shape
        grad = np.zeros((3, nx, ny, nz), dtype=field.dtype)
        
        for i in prange(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    grad[0, i, j, k] = (field[i+1, j, k] - field[i-1, j, k]) / (2 * dx)
                    grad[1, i, j, k] = (field[i, j+1, k] - field[i, j-1, k]) / (2 * dy)
                    grad[2, i, j, k] = (field[i, j, k+1] - field[i, j, k-1]) / (2 * dz)
        
        return grad
    
    @jit(nopython=True, parallel=True)
    def _compute_divergence_jit(vec_field: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
        """Compute divergence of vector field."""
        nx, ny, nz = vec_field.shape[1:]
        div = np.zeros((nx, ny, nz), dtype=vec_field.dtype)
        
        for i in prange(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    div[i, j, k] = (
                        (vec_field[0, i+1, j, k] - vec_field[0, i-1, j, k]) / (2 * dx) +
                        (vec_field[1, i, j+1, k] - vec_field[1, i, j-1, k]) / (2 * dy) +
                        (vec_field[2, i, j, k+1] - vec_field[2, i, j, k-1]) / (2 * dz)
                    )
        
        return div
    
    @jit(nopython=True, parallel=True)
    def _compute_laplacian_jit(field: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
        """Compute Laplacian using 7-point stencil."""
        nx, ny, nz = field.shape
        lap = np.zeros((nx, ny, nz), dtype=field.dtype)
        
        for i in prange(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    lap[i, j, k] = (
                        (field[i+1, j, k] + field[i-1, j, k]) / (dx * dx) +
                        (field[i, j+1, k] + field[i, j-1, k]) / (dy * dy) +
                        (field[i, j, k+1] + field[i, j, k-1]) / (dz * dz) -
                        6.0 * field[i, j, k] / (dx * dx)
                    )
        
        return lap
    
    @jit(nopython=True, parallel=True)
    def _compute_curl_jit(vec_field: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
        """Compute curl of vector field."""
        nx, ny, nz = vec_field.shape[1:]
        curl = np.zeros((3, nx, ny, nz), dtype=vec_field.dtype)
        
        for i in prange(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    curl[0, i, j, k] = (
                        (vec_field[2, i, j+1, k] - vec_field[2, i, j-1, k]) / (2 * dy) -
                        (vec_field[1, i, j, k+1] - vec_field[1, i, j, k-1]) / (2 * dz)
                    )
                    curl[1, i, j, k] = (
                        (vec_field[0, i, j, k+1] - vec_field[0, i, j, k-1]) / (2 * dz) -
                        (vec_field[2, i+1, j, k] - vec_field[2, i-1, j, k]) / (2 * dx)
                    )
                    curl[2, i, j, k] = (
                        (vec_field[1, i+1, j, k] - vec_field[1, i-1, j, k]) / (2 * dx) -
                        (vec_field[0, i, j+1, k] - vec_field[0, i, j-1, k]) / (2 * dy)
                    )
        
        return curl

else:
    # Fallback implementations without Numba
    def _compute_gradient_jit(field: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
        """Compute gradient with central differences (no Numba)."""
        grad = np.gradient(field, dx, dy, dz)
        return np.stack(grad)
    
    def _compute_divergence_jit(vec_field: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
        """Compute divergence (no Numba)."""
        return np.gradient(vec_field[0], axis=0) / dx + \
               np.gradient(vec_field[1], axis=1) / dy + \
               np.gradient(vec_field[2], axis=2) / dz
    
    def _compute_laplacian_jit(field: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
        """Compute Laplacian (no Numba)."""
        return (np.gradient(np.gradient(field, axis=0), axis=0) / (dx * dx) +
                np.gradient(np.gradient(field, axis=1), axis=1) / (dy * dy) +
                np.gradient(np.gradient(field, axis=2), axis=2) / (dz * dz))
    
    def _compute_curl_jit(vec_field: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
        """Compute curl (no Numba)."""
        dvy_dz = np.gradient(vec_field[1], axis=2) / dz
        dvz_dy = np.gradient(vec_field[2], axis=1) / dy
        dvz_dx = np.gradient(vec_field[2], axis=0) / dx
        dvx_dz = np.gradient(vec_field[0], axis=2) / dz
        dvy_dx = np.gradient(vec_field[1], axis=0) / dx
        dvx_dy = np.gradient(vec_field[0], axis=1) / dy
        
        curl_x = dvy_dz - dvz_dy
        curl_y = dvz_dx - dvx_dz
        curl_z = dvx_dy - dvy_dx
        return np.stack([curl_x, curl_y, curl_z])
