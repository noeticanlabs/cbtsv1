"""
Spectral Cache for HPC Optimization

Precomputes and caches k-space vectors, bin maps, FFT plans for reuse.

Author: CBTSV1 Team
Version: 1.0
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class SpectralCache:
    """
    Precomputed spectral data for FFT operations.
    
    Attributes:
        N: Grid dimensions (Nx, Ny, Nz)
        dx: Grid spacing
        kx, ky, kz: Wave vectors
        kx_bin, ky_bin, kz_bin: Bin indices for spectral analysis
        k_magnitude: |k| for each point
        dealias_mask: Mask for dealiasing (3/2 rule)
    """
    N: Tuple[int, int, int]
    dx: float
    kx: np.ndarray
    ky: np.ndarray
    kz: np.ndarray
    kx_bin: np.ndarray
    ky_bin: np.ndarray
    kz_bin: np.ndarray
    k_magnitude: np.ndarray
    dealias_mask: np.ndarray
    
    # Precomputed factors
    k2: np.ndarray  # |k|^2
    proj_factors: Dict[str, np.ndarray]  # Projection factors for various operators
    
    @classmethod
    def create(cls, N: Tuple[int, int, int], dx: float, n_bins: int = 3) -> 'SpectralCache':
        """Create a new spectral cache for the given grid."""
        Nx, Ny, Nz = N
        
        # Wave vectors (using rfftn convention for last axis)
        kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(Ny, d=dx) * 2 * np.pi
        kz = np.fft.rfftfreq(Nz, d=dx) * 2 * np.pi
        
        # Mesh
        kx_mesh, ky_mesh, kz_mesh = np.meshgrid(kx, ky, kz, indexing='ij')
        
        # Magnitude
        k_mag = np.sqrt(kx_mesh**2 + ky_mesh**2 + kz_mesh**2)
        
        # Bin mapping (3x3x3 = 27 bins)
        kx_min, kx_max = kx.min(), kx.max()
        ky_min, ky_max = ky.min(), ky.max()
        kz_min, kz_max = kz.min(), kz.max()
        
        kx_bin = np.clip(np.digitize(kx_mesh, np.linspace(kx_min, kx_max, n_bins + 1)) - 1, 0, n_bins - 1)
        ky_bin = np.clip(np.digitize(ky_mesh, np.linspace(ky_min, ky_max, n_bins + 1)) - 1, 0, n_bins - 1)
        kz_bin = np.clip(np.digitize(kz_mesh, np.linspace(kz_min, kz_max, n_bins + 1)) - 1, 0, n_bins - 1)
        
        # Dealias mask (3/2 rule: keep |k| < 2/3 * k_max)
        k_max = np.max(k_mag)
        dealias_mask = k_mag < (2.0 / 3.0) * k_max
        
        # Precompute k^2
        k2 = k_mag**2
        
        # Precompute projection factors
        proj_factors = {
            'laplacian': -k2,  # ∇² → -|k|²
            'gradient': 1j * kx_mesh,  # ∂_x → i*k_x (and similarly for y, z)
            'divergence': None,  # Computed per-field
            'advection': None,  # Computed per-field
        }
        
        return cls(
            N=N,
            dx=dx,
            kx=kx,
            ky=ky,
            kz=kz,
            kx_bin=kx_bin,
            ky_bin=ky_bin,
            kz_bin=kz_bin,
            k_magnitude=k_mag,
            dealias_mask=dealias_mask,
            k2=k2,
            proj_factors=proj_factors
        )
    
    def get_bin_mask(self, i: int, j: int, k: int) -> np.ndarray:
        """Get mask for a specific bin (i,j,k)."""
        return (self.kx_bin == i) & (self.ky_bin == j) & (self.kz_bin == k)
    
    def get_omega_bins(self) -> np.ndarray:
        """Compute omega values for each of the 27 bins."""
        omega = np.zeros(27)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    idx = i * 9 + j * 3 + k
                    mask = self.get_bin_mask(i, j, k)
                    if np.any(mask):
                        omega[idx] = np.sum(self.k_magnitude[mask])
        return omega


# Global cache instance (lazy initialization)
_SPECTRAL_CACHE: Optional[SpectralCache] = None


def get_spectral_cache(N: Tuple[int, int, int], dx: float) -> SpectralCache:
    """Get or create the global spectral cache."""
    global _SPECTRAL_CACHE
    
    if _SPECTRAL_CACHE is None or _SPECTRAL_CACHE.N != N:
        _SPECTRAL_CACHE = SpectralCache.create(N, dx)
    
    return _SPECTRAL_CACHE


def clear_spectral_cache():
    """Clear the global spectral cache."""
    global _SPECTRAL_CACHE
    _SPECTRAL_CACHE = None


class PreallocatedBufferManager:
    """
    Manages preallocated buffers for stencil operations.
    
    Eliminates allocation overhead in hot loops.
    """
    
    def __init__(self, shape: Tuple[int, int, int], dtype=np.float64):
        self.shape = shape
        self.dtype = dtype
        
        # Scratch buffers
        self.scratch_rhs_gamma = np.zeros(shape + (6,), dtype=dtype)
        self.scratch_rhs_K = np.zeros(shape + (6,), dtype=dtype)
        self.scratch_rhs_phi = np.zeros(shape, dtype=dtype)
        self.scratch_constraints = np.zeros(shape, dtype=dtype)
        
        # Halo buffers (for boundary handling)
        self.halo_gamma = np.zeros((6, 6) + shape[1:], dtype=dtype)  # 6 faces × 1-layer
        self.halo_K = np.zeros((6, 6) + shape[1:], dtype=dtype)
        
        # FFT work buffers
        self.fft_work = np.zeros(shape, dtype=np.complex128)
        
        # Diagnostic buffers
        self.norm_accum = np.zeros(10, dtype=dtype)
        self.max_accum = np.zeros(10, dtype=dtype)
    
    def get_rhs_buffer(self, name: str) -> np.ndarray:
        """Get a preallocated RHS buffer."""
        buffers = {
            'gamma': self.scratch_rhs_gamma,
            'K': self.scratch_rhs_K,
            'phi': self.scratch_rhs_phi,
        }
        return buffers.get(name, None)
    
    def get_constraint_buffer(self) -> np.ndarray:
        """Get constraint buffer."""
        return self.scratch_constraints
    
    def reset(self):
        """Reset all buffers to zero."""
        self.scratch_rhs_gamma.fill(0)
        self.scratch_rhs_K.fill(0)
        self.scratch_rhs_phi.fill(0)
        self.scratch_constraints.fill(0)
        self.fft_work.fill(0)
        self.norm_accum.fill(0)
        self.max_accum.fill(0)


# Per-grid buffer managers (lazy initialization)
_BUFFER_MANAGERS: Dict[Tuple[int, int, int], PreallocatedBufferManager] = {}


def get_buffer_manager(shape: Tuple[int, int, int], dtype=np.float64) -> PreallocatedBufferManager:
    """Get or create buffer manager for the given shape."""
    if shape not in _BUFFER_MANAGERS:
        _BUFFER_MANAGERS[shape] = PreallocatedBufferManager(shape, dtype)
    return _BUFFER_MANAGERS[shape]


def clear_buffer_managers():
    """Clear all buffer managers."""
    global _BUFFER_MANAGERS
    _BUFFER_MANAGERS = {}
