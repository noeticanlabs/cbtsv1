# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time"
]
LEXICON_SYMBOLS = {
    "\\\\Gamma": "GR_geom.christoffel",
    "R_{ij}": "GR_geom.ricci",
    "R": "GR_geom.scalar_curv"
}

import numpy as np
import hashlib
import collections
try:
    from numba import jit
except ImportError:
    jit = lambda f=None, **kwargs: f if f else (lambda g: g)
from .gr_core_fields import inv_sym6, sym6_to_mat33, mat33_to_sym6
from .gr_geometry_nsc import compute_christoffels_compiled, compute_ricci_compiled, second_covariant_derivative_scalar_compiled, lie_derivative_gamma_compiled, lie_derivative_K_compiled

def _fd_derivative_periodic(f, h, axis):
    """
    Computes the second-order centered finite difference with periodic boundary conditions.
    """
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2.0 * h)

@jit(nopython=True)
def _sym6_to_mat33_jit(sym6):
    """Numba-compatible sym6 to mat33."""
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
    """Numba-compatible inv_sym6."""
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
    """Hash a numpy array for caching purposes."""
    return hashlib.sha256(arr.tobytes()).hexdigest()

class GRGeometry:
    def __init__(self, fields):
        self.fields = fields
        Nx, Ny, Nz = fields.Nx, fields.Ny, fields.Nz
        # Preallocate geometry buffers
        self.christoffels = np.zeros((Nx, Ny, Nz, 3, 3, 3))
        self.Gamma = np.zeros((Nx, Ny, Nz, 3))
        self.ricci = np.zeros((Nx, Ny, Nz, 3, 3))
        self.R = np.zeros((Nx, Ny, Nz))
        # Scratch for ricci computation
        self.term3_scratch = np.zeros((Nx, Ny, Nz))
        self.term4_scratch = np.zeros((Nx, Ny, Nz))

        # Caching
        self._cache_maxsize = 32
        self._christoffel_cache = collections.OrderedDict()
        self._ricci_cache = collections.OrderedDict()
        self._ricci_for_metric_cache = collections.OrderedDict()
        self._cov_deriv_vec_cache = collections.OrderedDict()
        self._second_cov_deriv_scalar_cache = collections.OrderedDict()
        self._lie_gamma_cache = collections.OrderedDict()
        self._lie_K_cache = collections.OrderedDict()

    def clear_cache(self):
        """Clear all caches when fields are modified."""
        self._christoffel_cache.clear()
        self._ricci_cache.clear()
        self._ricci_for_metric_cache.clear()
        self._cov_deriv_vec_cache.clear()
        self._second_cov_deriv_scalar_cache.clear()
        self._lie_gamma_cache.clear()
        self._lie_K_cache.clear()

    def compute_christoffels(self):
        """Compute Christoffel symbols \\Gamma^k_{ij} and Gamma^i using compiled functions."""
        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz
        gamma = self.fields.gamma_sym6

        dgamma_dx = _fd_derivative_periodic(gamma, dx, axis=0)
        dgamma_dy = _fd_derivative_periodic(gamma, dy, axis=1)
        dgamma_dz = _fd_derivative_periodic(gamma, dz, axis=2)

        gamma_hash = hash_array(np.concatenate([gamma.flatten(), dgamma_dx.flatten(), dgamma_dy.flatten(), dgamma_dz.flatten()]))
        if gamma_hash in self._christoffel_cache:
            self.christoffels, self.Gamma = self._christoffel_cache[gamma_hash]
            self._christoffel_cache.move_to_end(gamma_hash)
            return

        self.christoffels, self.Gamma = compute_christoffels_compiled(gamma, dgamma_dx, dgamma_dy, dgamma_dz)

        # Cache the result
        self._christoffel_cache[gamma_hash] = (self.christoffels.copy(), self.Gamma.copy())
        if len(self._christoffel_cache) > self._cache_maxsize:
            self._christoffel_cache.popitem(last=False)

    def compute_ricci_for_metric(self, gamma_sym6, christoffels):
        """Compute Ricci tensor R_{ij} for a given metric and its Christoffels using compiled function."""
        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz
        
        d_christ_dx = _fd_derivative_periodic(christoffels, dx, axis=0)
        d_christ_dy = _fd_derivative_periodic(christoffels, dy, axis=1)
        d_christ_dz = _fd_derivative_periodic(christoffels, dz, axis=2)
        
        combined_hash = hash_array(np.concatenate([christoffels.flatten(), d_christ_dx.flatten(), d_christ_dy.flatten(), d_christ_dz.flatten()]))
        if combined_hash in self._ricci_for_metric_cache:
            return self._ricci_for_metric_cache[combined_hash]

        ricci = compute_ricci_compiled(christoffels, d_christ_dx, d_christ_dy, d_christ_dz)

        # Cache the result
        self._ricci_for_metric_cache[combined_hash] = ricci.copy()
        if len(self._ricci_for_metric_cache) > self._cache_maxsize:
            self._ricci_for_metric_cache.popitem(last=False)

        return ricci

    def compute_ricci(self):
        """Compute Ricci tensor R_{ij} using BSSN conformal decomposition."""
        combined_hash = hash_array(np.concatenate([self.fields.gamma_sym6.flatten(), self.fields.gamma_tilde_sym6.flatten(), self.fields.phi.flatten()]))
        if combined_hash in self._ricci_cache:
            self.ricci = self._ricci_cache[combined_hash]
            self._ricci_cache.move_to_end(combined_hash)
            return

        # Check for phi=0 case (flat conformal factor)
        chi = self.fields.phi
        if np.max(np.abs(chi)) < 1e-14:
            # If chi is zero, R_ij is just the Ricci tensor of gamma_sym6.
            # Use physical metric directly to avoid issues with inconsistent BSSN vars.
            self.compute_christoffels()
            self.ricci = self.compute_ricci_for_metric(self.fields.gamma_sym6, self.christoffels)
        else:
            # Compute \~{R}_{ij} using the conformal metric
            conformal_christoffels = np.zeros_like(self.christoffels)
            gamma_tilde = self.fields.gamma_tilde_sym6
            # Compute Christoffels for gamma_tilde
            self.compute_christoffels_for_metric(gamma_tilde, conformal_christoffels)
            R_tilde = self.compute_ricci_for_metric(gamma_tilde, conformal_christoffels)

            Nx, Ny, Nz = self.fields.Nx, self.fields.Ny, self.fields.Nz
            dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz

            # grad_chi
            grad_chi = np.zeros((Nx, Ny, Nz, 3))
            grad_chi[..., 0] = _fd_derivative_periodic(chi, dx, axis=0)
            grad_chi[..., 1] = _fd_derivative_periodic(chi, dy, axis=1)
            grad_chi[..., 2] = _fd_derivative_periodic(chi, dz, axis=2)

            # DD_chi = D_i D_j chi
            DD_chi = self.second_covariant_derivative_scalar(chi, christoffels=conformal_christoffels)

            # Lap_chi = gamma^{ij} D_i D_j chi
            gamma_tilde_inv = inv_sym6(gamma_tilde)
            gamma_tilde_inv_full = sym6_to_mat33(gamma_tilde_inv)
            Lap_chi = np.einsum('...ij,...ij', gamma_tilde_inv_full, DD_chi)

            # D^k chi D_k chi
            D_chi_D_chi = np.einsum('...ij,...i,...j', gamma_tilde_inv_full, grad_chi, grad_chi)

            gamma_tilde_full = sym6_to_mat33(gamma_tilde)

            R_chi = (-2 * DD_chi
                     - 2 * gamma_tilde_full * Lap_chi[..., np.newaxis, np.newaxis]
                     + 4 * np.einsum('...i,...j->...ij', grad_chi, grad_chi)
                     - 4 * gamma_tilde_full * D_chi_D_chi[..., np.newaxis, np.newaxis])

            self.ricci = R_tilde + R_chi

        # Cache the result
        self._ricci_cache[combined_hash] = self.ricci.copy()
        if len(self._ricci_cache) > self._cache_maxsize:
            self._ricci_cache.popitem(last=False)

    def compute_christoffels_for_metric(self, gamma_sym6, christoffels_out):
        """Compute Christoffel symbols for a given metric."""
        Nx, Ny, Nz = self.fields.Nx, self.fields.Ny, self.fields.Nz
        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz

        dgamma_dx = _fd_derivative_periodic(gamma_sym6, dx, axis=0)
        dgamma_dy = _fd_derivative_periodic(gamma_sym6, dy, axis=1)
        dgamma_dz = _fd_derivative_periodic(gamma_sym6, dz, axis=2)

        christoffels_out[...], _ = compute_christoffels_compiled(gamma_sym6, dgamma_dx, dgamma_dy, dgamma_dz)

    def compute_scalar_curvature(self):
        """Compute scalar curvature R = gamma^{ij} R_{ij}."""
        if not hasattr(self, 'ricci') or self.ricci is None:
            self.compute_ricci()

        gamma_inv = inv_sym6(self.fields.gamma_sym6)
        gamma_inv_full = sym6_to_mat33(gamma_inv)

        self.R = np.einsum('...ij,...ij', gamma_inv_full, self.ricci)

    def covariant_derivative_vector(self, V):
        """Compute covariant derivative D_k V^i = ∂_k V^i + Γ^i_{jk} V^j"""
        combined_hash = hash_array(np.concatenate([self.fields.gamma_sym6.flatten(), V.flatten()]))
        if combined_hash in self._cov_deriv_vec_cache:
            return self._cov_deriv_vec_cache[combined_hash]

        if not hasattr(self, 'christoffels') or self.christoffels is None:
            self.compute_christoffels()

        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz
        Nx, Ny, Nz = self.fields.Nx, self.fields.Ny, self.fields.Nz

        grad_V = np.zeros((Nx, Ny, Nz, 3, 3))
        for i in range(3):
            grad_V[..., 0, i] = _fd_derivative_periodic(V[..., i], dx, axis=0)
            grad_V[..., 1, i] = _fd_derivative_periodic(V[..., i], dy, axis=1)
            grad_V[..., 2, i] = _fd_derivative_periodic(V[..., i], dz, axis=2)

        D_V = grad_V + np.einsum('...ijk,...j->...ki', self.christoffels, V)

        self._cov_deriv_vec_cache[combined_hash] = D_V.copy()
        if len(self._cov_deriv_vec_cache) > self._cache_maxsize:
            self._cov_deriv_vec_cache.popitem(last=False)

        return D_V

    def second_covariant_derivative_scalar(self, scalar, christoffels=None):
        """Compute D_i D_j scalar = ∂_i ∂_j scalar - Γ^k_{ij} ∂_k scalar using compiled function"""
        if christoffels is None:
            if not hasattr(self, 'christoffels') or self.christoffels is None:
                self.compute_christoffels()
            christoffels_to_use = self.christoffels
            gamma_hash = hash_array(self.fields.gamma_sym6)
        else:
            christoffels_to_use = christoffels
            gamma_hash = hash_array(christoffels)
        combined_hash = gamma_hash + '_' + hash_array(scalar.flatten())
        if combined_hash in self._second_cov_deriv_scalar_cache:
            return self._second_cov_deriv_scalar_cache[combined_hash]

        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz

        DD_scalar = second_covariant_derivative_scalar_compiled(scalar, christoffels_to_use, dx, dy, dz)

        self._second_cov_deriv_scalar_cache[combined_hash] = DD_scalar.copy()
        if len(self._second_cov_deriv_scalar_cache) > self._cache_maxsize:
            self._second_cov_deriv_scalar_cache.popitem(last=False)

        return DD_scalar

    def lie_derivative_gamma(self, gamma_sym6, beta):
        """Compute Lie derivative L_β γ_ij using compiled function"""
        combined_hash = hash_array(np.concatenate([gamma_sym6.flatten(), beta.flatten()]))
        if combined_hash in self._lie_gamma_cache:
            return self._lie_gamma_cache[combined_hash]

        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz

        lie_gamma_sym6 = lie_derivative_gamma_compiled(gamma_sym6, beta, dx, dy, dz)

        self._lie_gamma_cache[combined_hash] = lie_gamma_sym6.copy()
        if len(self._lie_gamma_cache) > self._cache_maxsize:
            self._lie_gamma_cache.popitem(last=False)

        return lie_gamma_sym6

    def lie_derivative_K(self, K_sym6, beta):
        """Compute Lie derivative L_β K_ij using compiled function"""
        combined_hash = hash_array(np.concatenate([K_sym6.flatten(), beta.flatten()]))
        if combined_hash in self._lie_K_cache:
            return self._lie_K_cache[combined_hash]

        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz

        lie_K_sym6 = lie_derivative_K_compiled(K_sym6, beta, dx, dy, dz)

        self._lie_K_cache[combined_hash] = lie_K_sym6.copy()
        if len(self._lie_K_cache) > self._cache_maxsize:
            self._lie_K_cache.popitem(last=False)

        return lie_K_sym6

    def enforce_det_gamma_tilde(self):
        """Enforce det(\tilde{\gamma}) = 1 by rescaling \tilde{\gamma}."""
        gamma_tilde_full = sym6_to_mat33(self.fields.gamma_tilde_sym6)
        det_gamma_tilde = np.linalg.det(gamma_tilde_full)
        det_third_root = det_gamma_tilde ** (1.0 / 3.0)
        gamma_tilde_full_corrected = gamma_tilde_full / det_third_root[..., np.newaxis, np.newaxis]
        self.fields.gamma_tilde_sym6 = mat33_to_sym6(gamma_tilde_full_corrected)
        self.clear_cache()

    def enforce_traceless_A(self):
        """Enforce traceless A: A_ij = A_ij - (1/3) \tilde{\gamma}_{ij} (\tilde{\gamma}^{kl} A_kl)."""
        gamma_tilde_inv = inv_sym6(self.fields.gamma_tilde_sym6)
        gamma_tilde_inv_full = sym6_to_mat33(gamma_tilde_inv)
        gamma_tilde_full = sym6_to_mat33(self.fields.gamma_tilde_sym6)
        A_full = sym6_to_mat33(self.fields.A_sym6)
        trace_A = np.einsum('...kl,...kl', gamma_tilde_inv_full, A_full)
        correction = (1.0 / 3.0) * gamma_tilde_full * trace_A[..., np.newaxis, np.newaxis]
        A_full_corrected = A_full - correction
        self.fields.A_sym6 = mat33_to_sym6(A_full_corrected)

    def compute_constraint_proxy(self, scale: str) -> float:
        """Compute proxy constraint residual for a given scale."""
        if not hasattr(self, 'R') or self.R is None:
            self.compute_scalar_curvature()

        if scale == 'L':
            return np.abs(self.R).mean()
        elif scale == 'M':
            return np.abs(np.gradient(self.R)).mean()
        else:  # 'H'
            return np.abs(self.R).max()

    def compute_all(self):
        """Compute all geometric quantities."""
        self.compute_christoffels()
        self.compute_ricci()
        self.compute_scalar_curvature()

def ricci_tensor_kernel(fields):
    """Standalone kernel for computing Ricci tensor."""
    geometry = GRGeometry(fields)
    geometry.compute_ricci()
    return geometry.ricci


def connection_coeff(lambda_idx, mu_idx, nu_idx, metric, coords):
    """
    Compute Christoffel symbol of the second kind: Γ^λ_μν
    
    Args:
        lambda_idx: Upper index λ (spatial direction 0,1,2)
        mu_idx: Lower index μ (spatial direction 0,1,2)
        nu_idx: Lower index ν (spatial direction 0,1,2)
        metric: Metric tensor g_ij in sym6 format [Nx, Ny, Nz, 6]
        coords: Coordinate array [Nx, Ny, Nz, 3] containing (x, y, z)
    
    Returns:
        Christoffel symbol Γ^λ_μν at each grid point [Nx, Ny, Nz]
        
    Formula:
        Γ^λ_μν = ½g^λσ(∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν)
    """
    from src.core.gr_core_fields import inv_sym6, sym6_to_mat33
    
    Nx, Ny, Nz = metric.shape[:3]
    dx = coords[1,0,0,0] - coords[0,0,0,0] if Nx > 1 else 1.0
    dy = coords[0,1,0,1] - coords[0,0,0,1] if Ny > 1 else 1.0
    dz = coords[0,0,1,2] - coords[0,0,0,2] if Nz > 1 else 1.0
    
    # Compute inverse metric g^ij
    gamma_inv = inv_sym6(metric)
    
    # Convert to full 3x3 matrices for easier access
    gamma_inv_full = sym6_to_mat33(gamma_inv)
    
    # Compute partial derivatives of metric components
    # dgamma[..., i, j, alpha] = ∂_alpha g_ij where alpha is derivative direction (0=x, 1=y, 2=z)
    # Shape: (Nx, Ny, Nz, 3, 3, 3)
    dgamma = np.zeros((Nx, Ny, Nz, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            # Map (i,j) to sym6 index
            if i <= j:
                sym6_idx = i * 3 - i * (i + 1) // 2 + j
            else:
                sym6_idx = j * 3 - j * (j + 1) // 2 + i
            # dgamma[..., i, j, alpha] where alpha is the derivative direction
            dgamma[..., i, j, 0] = _fd_derivative_periodic(metric[..., sym6_idx], dx, axis=0)
            dgamma[..., i, j, 1] = _fd_derivative_periodic(metric[..., sym6_idx], dy, axis=1)
            dgamma[..., i, j, 2] = _fd_derivative_periodic(metric[..., sym6_idx], dz, axis=2)
    
    # Christoffel symbol computation: Γ^λ_μν = ½g^λσ(∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν)
    christoffel = np.zeros((Nx, Ny, Nz))
    
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                result = 0.0
                for sigma in range(3):
                    # g^λσ
                    g_upper = gamma_inv_full[i, j, k, lambda_idx, sigma]
                    
                    # ∂_μ g_νσ (derivative along mu_idx direction of nu-sigma component)
                    d_g_nusigma = dgamma[i, j, k, nu_idx, sigma, mu_idx]
                    
                    # ∂_ν g_μσ (derivative along nu_idx direction of mu-sigma component)
                    d_g_musigma = dgamma[i, j, k, mu_idx, sigma, nu_idx]
                    
                    # ∂_σ g_μν (derivative along sigma direction of mu-nu component)
                    d_g_munu = dgamma[i, j, k, mu_idx, nu_idx, sigma]
                    
                    result += 0.5 * g_upper * (d_g_nusigma + d_g_musigma - d_g_munu)
                
                christoffel[i, j, k] = result
    
    return christoffel


def lambda_laplacian(field, lambda_param, coords, metric):
    """
    Compute covariant Laplacian with coupling constant: ∇_λ∇^λ φ + λ² φ
    
    Args:
        field: Scalar field φ [Nx, Ny, Nz]
        lambda_param: Coupling constant λ (scalar)
        coords: Coordinate array [Nx, Ny, Nz, 3] containing (x, y, z)
        metric: Metric tensor g_ij in sym6 format [Nx, Ny, Nz, 6]
    
    Returns:
        □_λφ = g^μν ∇_μ ∇_ν φ + λ² φ at each grid point [Nx, Ny, Nz]
        
    The covariant Laplacian on a curved manifold:
        □_λφ = g^μν (∂_μ ∂_ν φ - Γ^σ_μν ∂_σ φ) + λ² φ
    """
    from src.core.gr_core_fields import inv_sym6, sym6_to_mat33
    
    Nx, Ny, Nz = metric.shape[:3]
    dx = coords[1,0,0,0] - coords[0,0,0,0] if Nx > 1 else 1.0
    dy = coords[0,1,0,1] - coords[0,0,0,1] if Ny > 1 else 1.0
    dz = coords[0,0,1,2] - coords[0,0,0,2] if Nz > 1 else 1.0
    
    # Compute inverse metric g^ij
    gamma_inv = inv_sym6(metric)
    gamma_inv_full = sym6_to_mat33(gamma_inv)
    
    # Compute Christoffel symbols first using the GRGeometry class
    class FieldsStub:
        def __init__(self, Nx, Ny, Nz, dx, dy, dz):
            self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
            self.dx, self.dy, self.dz = dx, dy, dz
            self.gamma_sym6 = metric
    
    fields_stub = FieldsStub(Nx, Ny, Nz, dx, dy, dz)
    geom = GRGeometry(fields_stub)
    geom.compute_christoffels()
    christoffels = geom.christoffels
    
    # Compute first covariant derivative: ∇_μ φ = ∂_μ φ
    grad_phi = np.zeros((Nx, Ny, Nz, 3))
    grad_phi[..., 0] = _fd_derivative_periodic(field, dx, axis=0)
    grad_phi[..., 1] = _fd_derivative_periodic(field, dy, axis=1)
    grad_phi[..., 2] = _fd_derivative_periodic(field, dz, axis=2)
    
    # Compute second covariant derivative: ∇_ν ∇_μ φ = ∂_ν ∂_μ φ - Γ^σ_μν ∂_σ φ
    DD_phi = np.zeros((Nx, Ny, Nz, 3, 3))
    for mu in range(3):
        for nu in range(3):
            # ∂_ν ∂_μ φ
            if nu == 0:
                d_dmu = _fd_derivative_periodic(grad_phi[..., mu], dx, axis=0)
            elif nu == 1:
                d_dmu = _fd_derivative_periodic(grad_phi[..., mu], dy, axis=1)
            else:
                d_dmu = _fd_derivative_periodic(grad_phi[..., mu], dz, axis=2)
            
            # -Γ^σ_μν ∂_σ φ (sum over sigma)
            christoffel_term = np.zeros((Nx, Ny, Nz))
            for sigma in range(3):
                christoffel_term += christoffels[..., sigma, mu, nu] * grad_phi[..., sigma]
            
            DD_phi[..., mu, nu] = d_dmu - christoffel_term
    
    # Compute covariant Laplacian: g^μν ∇_μ ∇_ν φ
    laplacian = np.einsum('...ij,...ij', gamma_inv_full, DD_phi)
    
    # Add coupling term: λ² φ
    lambda_sq = lambda_param ** 2
    result = laplacian + lambda_sq * field
    
    return result
