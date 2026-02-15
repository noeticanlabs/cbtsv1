# gr_geometry.py
# =============================================================================
# GR Geometry Computation Module with Enhanced Solvers Integration
# =============================================================================
#
# Enhanced with CompactFD 4th order finite difference schemes.

import numpy as np
import hashlib
import collections
try:
    from numba import jit
except ImportError:
    jit = lambda f=None, **kwargs: f if f else (lambda g: g)
from .gr_core_fields import inv_sym6, sym6_to_mat33, mat33_to_sym6
from .gr_geometry_nsc import compute_christoffels_compiled, compute_ricci_compiled, second_covariant_derivative_scalar_compiled, lie_derivative_gamma_compiled, lie_derivative_K_compiled


# ============================================================================
# ARRAY BOUNDS VALIDATION UTILITY
# ============================================================================

def _validate_array_size(arr: np.ndarray, min_size: int, name: str = "array"):
    """
    Validate array has minimum required size.
    
    Use this before accessing arr[-n] to ensure the array has enough elements.
    For example, before using arr[-2], arr[-3], call with min_size equal to
    the maximum negative index (2, 3, etc.).
    
    Args:
        arr: Array to validate
        min_size: Minimum required size (0-indexed, so min_size=4 allows arr[-4])
        name: Name of array for error messages
        
    Raises:
        ValueError: If array is smaller than min_size
    """
    if arr.size < min_size:
        raise ValueError(
            f"{name} has insufficient size: {arr.size} < {min_size} required. "
            f"Shape: {arr.shape}. Cannot access index -{min_size} safely."
        )


# ============================================================================
# COMPACT FINITE DIFFERENCE SCHEMES
# ============================================================================

class CompactFD:
    """
    4th order compact finite difference scheme for derivatives.
    
    Scheme: (1/4)*f'_{i-1} + f'_i + (1/4)*f'_{i+1} = (3/(2h)) * (f_{i+1} - f_{i-1})
    
    Solved using Thomas algorithm for tridiagonal systems.
    """
    
    def __init__(self, order: int = 4):
        if order != 4:
            raise ValueError("Only order=4 supported")
        self.alpha = 0.25  # 1/4
    
    def first_derivative(self, f: np.ndarray, h: float) -> np.ndarray:
        """
        Compute first derivative using 4th order compact Pade scheme.
        
        The tridiagonal system is:
        alpha * f'_{i-1} + f'_i + alpha * f'_{i+1} = rhs_i
        
        where rhs_i = (3/(2h)) * (f_{i+1} - f_{i-1})
        
        Raises:
            ValueError: If array is too small for stencil
        """
        n = len(f)
        # Validate minimum size: need at least 2 elements for f[1], f[0], f[-2], f[-1]
        _validate_array_size(f, 2, "f (first_derivative)")
        
        alpha = self.alpha
        
        # Build RHS
        rhs = np.zeros(n)
        rhs[0] = (f[1] - f[0]) / h  # Forward
        rhs[-1] = (f[-1] - f[-2]) / h  # Backward
        
        # Interior points
        rhs[1:-1] = (3.0 / (2.0 * h)) * (f[2:] - f[:-2])
        
        # Tridiagonal solver: alpha * x[i-1] + x[i] + alpha * x[i+1] = rhs[i]
        # Thomas algorithm
        a = alpha * np.ones(n)  # sub-diagonal
        b = np.ones(n)  # diagonal
        c = alpha * np.ones(n)  # super-diagonal
        
        # Forward sweep
        cp = np.zeros(n)
        dp = np.zeros(n)
        
        cp[0] = c[0] / b[0]
        dp[0] = rhs[0] / b[0]
        
        for i in range(1, n):
            denom = b[i] - a[i] * cp[i-1]
            if i < n - 1:
                cp[i] = c[i] / denom
            dp[i] = (rhs[i] - a[i] * dp[i-1]) / denom
        
        # Back substitution
        df = np.zeros(n)
        df[-1] = dp[-1]
        
        for i in range(n-2, -1, -1):
            df[i] = dp[i] - cp[i] * df[i+1]
        
        return df


# ============================================================================
# LEGACY FUNCTIONS
# ============================================================================

def _fd_derivative_periodic(f, h, axis):
    """2nd order centered finite difference."""
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2.0 * h)


# ============================================================================
# GR GEOMETRY CLASS
# ============================================================================

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

def hash_array(arr):
    return hashlib.sha256(arr.tobytes()).hexdigest()


class GRGeometry:
    """
    Geometry Computer for GR Evolution.
    
    Supports 'central2' (2nd order) and 'compact4' (4th order) FD schemes.
    """
    
    def __init__(self, fields, fd_method='central2'):
        self.fields = fields
        Nx, Ny, Nz = fields.Nx, fields.Ny, fields.Nz
        
        self.christoffels = np.zeros((Nx, Ny, Nz, 3, 3, 3))
        self.Gamma = np.zeros((Nx, Ny, Nz, 3))
        self.ricci = np.zeros((Nx, Ny, Nz, 3, 3))
        self.R = np.zeros((Nx, Ny, Nz))
        
        self.fd_method = fd_method
        if fd_method == 'compact4':
            self.compact_x = CompactFD(order=4)
            self.compact_y = CompactFD(order=4)
            self.compact_z = CompactFD(order=4)
        
        # Caching
        self._cache_maxsize = 32
        self._christoffel_cache = collections.OrderedDict()
        self._ricci_cache = collections.OrderedDict()
        self._cov_deriv_vec_cache = collections.OrderedDict()
        self._second_cov_deriv_scalar_cache = collections.OrderedDict()
        self._lie_gamma_cache = collections.OrderedDict()
        self._lie_K_cache = collections.OrderedDict()
    
    def clear_cache(self):
        self._christoffel_cache.clear()
        self._ricci_cache.clear()
        self._cov_deriv_vec_cache.clear()
        self._second_cov_deriv_scalar_cache.clear()
        self._lie_gamma_cache.clear()
        self._lie_K_cache.clear()
    
    def compute_christoffels(self):
        """Compute Christoffel symbols."""
        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz
        gamma = self.fields.gamma_sym6
        
        # Use configured FD method
        if self.fd_method == 'central2':
            dgamma_dx = _fd_derivative_periodic(gamma, dx, 0)
            dgamma_dy = _fd_derivative_periodic(gamma, dy, 1)
            dgamma_dz = _fd_derivative_periodic(gamma, dz, 2)
        else:  # compact4
            # Apply compact FD to each component
            dgamma_dx = np.zeros_like(gamma)
            dgamma_dy = np.zeros_like(gamma)
            dgamma_dz = np.zeros_like(gamma)
            
            for i in range(gamma.shape[-1]):
                for j in range(dx.shape[0]):
                    for k in range(dx.shape[1]):
                        for l in range(dx.shape[2]):
                            dgamma_dx[j,k,l,i] = self.compact_x.first_derivative(gamma[j,k,:,i], dz)
                            dgamma_dy[j,k,l,i] = self.compact_y.first_derivative(gamma[j,:,l,i], dy)
                            dgamma_dz[:,j,k,i] = self.compact_z.first_derivative(gamma[:,j,k,i], dx)
        
        gamma_hash = hash_array(np.concatenate([gamma.flatten(), dgamma_dx.flatten(), dgamma_dy.flatten(), dgamma_dz.flatten()]))
        if gamma_hash in self._christoffel_cache:
            self.christoffels, self.Gamma = self._christoffel_cache[gamma_hash]
            self._christoffel_cache.move_to_end(gamma_hash)
            return
        
        self.christoffels, self.Gamma = compute_christoffels_compiled(gamma, dgamma_dx, dgamma_dy, dgamma_dz)
        
        self._christoffel_cache[gamma_hash] = (self.christoffels.copy(), self.Gamma.copy())
        if len(self._christoffel_cache) > self._cache_maxsize:
            self._christoffel_cache.popitem(last=False)
    
    def compute_ricci(self):
        """Compute Ricci tensor."""
        combined_hash = hash_array(np.concatenate([self.fields.gamma_sym6.flatten(), self.fields.phi.flatten()]))
        if combined_hash in self._ricci_cache:
            self.ricci = self._ricci_cache[combined_hash]
            self._ricci_cache.move_to_end(combined_hash)
            return
        
        self.compute_christoffels()
        self.ricci = self.compute_ricci_for_metric(self.fields.gamma_sym6, self.christoffels)
        
        self._ricci_cache[combined_hash] = self.ricci.copy()
        if len(self._ricci_cache) > self._cache_maxsize:
            self._ricci_cache.popitem(last=False)
    
    def compute_ricci_for_metric(self, gamma_sym6, christoffels):
        """Compute Ricci for given metric."""
        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz
        d_christ_dx = _fd_derivative_periodic(christoffels, dx, axis=0)
        d_christ_dy = _fd_derivative_periodic(christoffels, dy, axis=1)
        d_christ_dz = _fd_derivative_periodic(christoffels, dz, axis=2)
        return compute_ricci_compiled(christoffels, d_christ_dx, d_christ_dy, d_christ_dz)
    
    def compute_scalar_curvature(self):
        """Compute scalar curvature."""
        if not hasattr(self, 'ricci') or self.ricci is None:
            self.compute_ricci()
        gamma_inv = inv_sym6(self.fields.gamma_sym6)
        gamma_inv_full = sym6_to_mat33(gamma_inv)
        self.R = np.einsum('...ij,...ij', gamma_inv_full, self.ricci)
    
    def compute_all(self):
        """Compute all geometric quantities."""
        self.compute_christoffels()
        self.compute_ricci()
        self.compute_scalar_curvature()

    def lie_derivative_gamma(self, gamma_sym6, beta):
        """
        Compute Lie derivative of the metric tensor gamma_ij.
        
        L_β γ_ij = β^k ∂_k γ_ij + γ_ik ∂_j β^k + γ_jk ∂_i β^k
        
        Args:
            gamma_sym6: Metric tensor in symmetry-6 format (Nx, Ny, Nz, 6)
            beta: Shift vector (Nx, Ny, Nz, 3)
            
        Returns:
            Lie derivative in symmetry-6 format (Nx, Ny, Nz, 6)
        """
        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz
        return lie_derivative_gamma_compiled(gamma_sym6, beta, dx, dy, dz)

    def lie_derivative_K(self, K_sym6, beta):
        """
        Compute Lie derivative of the extrinsic curvature K_ij.
        
        L_β K_ij = β^k ∂_k K_ij + K_ik ∂_j β^k + K_jk ∂_i β^k
        
        Args:
            K_sym6: Extrinsic curvature in symmetry-6 format (Nx, Ny, Nz, 6)
            beta: Shift vector (Nx, Ny, Nz, 3)
            
        Returns:
            Lie derivative in symmetry-6 format (Nx, Ny, Nz, 6)
        """
        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz
        return lie_derivative_K_compiled(K_sym6, beta, dx, dy, dz)

    def second_covariant_derivative_scalar(self, scalar_field):
        """
        Compute second covariant derivative of a scalar field.
        
        ∇_i ∇_j f = ∂_i ∂_j f - Γ^k_ij ∂_k f
        
        Args:
            scalar_field: Scalar field (Nx, Ny, Nz)
            
        Returns:
            Second covariant derivative tensor (Nx, Ny, Nz, 3, 3)
        """
        # Ensure Christoffel symbols are computed
        self.compute_christoffels()
        
        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz
        return second_covariant_derivative_scalar_compiled(
            scalar_field, self.christoffels, dx, dy, dz
        )

    def compute_christoffels_for_metric(self, metric_sym6, christoffels_output):
        """
        Compute Christoffel symbols for a given metric (not stored in self.fields).
        
        Args:
            metric_sym6: Metric tensor in symmetry-6 format (Nx, Ny, Nz, 6)
            christoffels_output: Preallocated array for output (Nx, Ny, Nz, 3, 3, 3)
        """
        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz
        
        # Compute derivatives of the given metric
        if self.fd_method == 'central2':
            dmetric_dx = _fd_derivative_periodic(metric_sym6, dx, 0)
            dmetric_dy = _fd_derivative_periodic(metric_sym6, dy, 1)
            dmetric_dz = _fd_derivative_periodic(metric_sym6, dz, 2)
        else:
            raise NotImplementedError(
                f"compute_christoffels_for_metric not implemented for fd_method={self.fd_method}"
            )
        
        # Compute Christoffel symbols using the compiled function
        christoffels, _ = compute_christoffels_compiled(
            metric_sym6, dmetric_dx, dmetric_dy, dmetric_dz
        )
        christoffels_output[:] = christoffels

    def enforce_det_gamma_tilde(self):
        """
        Enforce det(gamma_tilde) = 1 constraint on the conformal metric.
        
        This ensures the conformal metric remains in the proper gauge.
        Uses Wilsonian renormalization approach.
        """
        if not hasattr(self.fields, 'gamma_tilde_sym6'):
            return  # Nothing to enforce
            
        gamma_tilde = self.fields.gamma_tilde_sym6
        
        # Compute current determinant
        xx = gamma_tilde[..., 0]
        yy = gamma_tilde[..., 3]
        zz = gamma_tilde[..., 5]
        xy = gamma_tilde[..., 1]
        xz = gamma_tilde[..., 2]
        yz = gamma_tilde[..., 4]
        
        det = xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz) + xz * (xy * yz - yy * xz)
        
        # Wilsonian renormalization: scale to enforce det = 1
        # gamma_tilde_new = gamma_tilde * (1/det)^(1/3)
        factor = np.power(det, -1.0 / 3.0)
        
        # Handle NaN/Inf
        factor = np.where(np.isfinite(factor), factor, 1.0)
        
        self.fields.gamma_tilde_sym6[..., 0] = xx * factor
        self.fields.gamma_tilde_sym6[..., 1] = xy * factor
        self.fields.gamma_tilde_sym6[..., 2] = xz * factor
        self.fields.gamma_tilde_sym6[..., 3] = yy * factor
        self.fields.gamma_tilde_sym6[..., 4] = yz * factor
        self.fields.gamma_tilde_sym6[..., 5] = zz * factor

    def enforce_traceless_A(self):
        """
        Enforce tr(A) = 0 constraint on the traceless extrinsic curvature.
        
        This ensures A_ij remains traceless: A^k_k = 0
        """
        if not hasattr(self.fields, 'A_sym6'):
            return  # Nothing to enforce
            
        if not hasattr(self.fields, 'gamma_sym6'):
            return  # Need metric to compute trace
            
        A = self.fields.A_sym6
        gamma = self.fields.gamma_sym6
        
        # Compute gamma inverse
        gamma_inv = inv_sym6(gamma)
        
        # Compute trace: A^k_k = gamma^{ij} A_ij
        # Convert to full 3x3 and compute trace
        A_full = np.zeros(gamma.shape[:-1] + (3, 3), dtype=A.dtype)
        A_full[..., 0, 0] = A[..., 0]
        A_full[..., 0, 1] = A[..., 1]
        A_full[..., 0, 2] = A[..., 2]
        A_full[..., 1, 0] = A[..., 1]
        A_full[..., 1, 1] = A[..., 3]
        A_full[..., 1, 2] = A[..., 4]
        A_full[..., 2, 0] = A[..., 2]
        A_full[..., 2, 1] = A[..., 4]
        A_full[..., 2, 2] = A[..., 5]
        
        gamma_inv_full = np.zeros(gamma.shape[:-1] + (3, 3), dtype=gamma.dtype)
        gamma_inv_full[..., 0, 0] = gamma[..., 0]
        gamma_inv_full[..., 0, 1] = gamma[..., 1]
        gamma_inv_full[..., 0, 2] = gamma[..., 2]
        gamma_inv_full[..., 1, 0] = gamma[..., 1]
        gamma_inv_full[..., 1, 1] = gamma[..., 3]
        gamma_inv_full[..., 1, 2] = gamma[..., 4]
        gamma_inv_full[..., 2, 0] = gamma[..., 2]
        gamma_inv_full[..., 2, 1] = gamma[..., 4]
        gamma_inv_full[..., 2, 2] = gamma[..., 5]
        
        # Compute trace
        trace_A = np.einsum('...ij,...ij->...', A_full, gamma_inv_full)
        
        # Subtract trace/3 from diagonal to make traceless
        # (A_ij - (1/3) gamma_ij A^k_k) in 3D
        trace_part = trace_A / 3.0
        
        self.fields.A_sym6[..., 0] = A[..., 0] - trace_part * gamma[..., 0]
        self.fields.A_sym6[..., 1] = A[..., 1] - trace_part * gamma[..., 1]
        self.fields.A_sym6[..., 2] = A[..., 2] - trace_part * gamma[..., 2]
        self.fields.A_sym6[..., 3] = A[..., 3] - trace_part * gamma[..., 3]
        self.fields.A_sym6[..., 4] = A[..., 4] - trace_part * gamma[..., 4]
        self.fields.A_sym6[..., 5] = A[..., 5] - trace_part * gamma[..., 5]


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_geometry(fields, method='central2'):
    """Factory function to create GRGeometry with specified FD method."""
    return GRGeometry(fields, fd_method=method)


# ============================================================================
# STANDALONE FUNCTIONS
# ============================================================================

def ricci_tensor_kernel(fields):
    """Compute Ricci tensor."""
    geometry = GRGeometry(fields)
    geometry.compute_ricci()
    return geometry.ricci


def connection_coeff(lambda_idx, mu_idx, nu_idx, metric, coords):
    """Compute Christoffel symbol."""
    Nx, Ny, Nz = metric.shape[:3]
    dx = coords[1,0,0,0] - coords[0,0,0,0] if Nx > 1 else 1.0
    dy = coords[0,1,0,1] - coords[0,0,0,1] if Ny > 1 else 1.0
    dz = coords[0,0,1,2] - coords[0,0,0,2] if Nz > 1 else 1.0
    
    gamma_inv = inv_sym6(metric)
    gamma_inv_full = sym6_to_mat33(gamma_inv)
    
    dgamma = np.zeros((Nx, Ny, Nz, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            if i <= j:
                sym6_idx = i * 3 - i * (i + 1) // 2 + j
            else:
                sym6_idx = j * 3 - j * (j + 1) // 2 + i
            dgamma[..., i, j, 0] = _fd_derivative_periodic(metric[..., sym6_idx], dx, axis=0)
            dgamma[..., i, j, 1] = _fd_derivative_periodic(metric[..., sym6_idx], dy, axis=1)
            dgamma[..., i, j, 2] = _fd_derivative_periodic(metric[..., sym6_idx], dz, axis=2)
    
    christoffel = np.zeros((Nx, Ny, Nz))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                result = 0.0
                for sigma in range(3):
                    g_upper = gamma_inv_full[i, j, k, lambda_idx, sigma]
                    d_g_nusigma = dgamma[i, j, k, nu_idx, sigma, mu_idx]
                    d_g_musigma = dgamma[i, j, k, mu_idx, sigma, nu_idx]
                    d_g_munu = dgamma[i, j, k, mu_idx, nu_idx, sigma]
                    result += 0.5 * g_upper * (d_g_nusigma + d_g_musigma - d_g_munu)
                christoffel[i, j, k] = result
    
    return christoffel
