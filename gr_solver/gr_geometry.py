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
from gr_geometry_nsc import compute_christoffels_compiled, compute_ricci_compiled, second_covariant_derivative_scalar_compiled, lie_derivative_gamma_compiled, lie_derivative_K_compiled

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

@jit(nopython=True)
def _compute_christoffels_jit(Nx, Ny, Nz, gamma_sym6, dx, dy, dz):
    """JIT-compiled computation of Christoffel symbols."""
    # Convert to full 3x3 tensor for derivatives
    gamma_full = np.zeros((Nx, Ny, Nz, 3, 3))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                gamma_full[i,j,k] = _sym6_to_mat33_jit(gamma_sym6[i,j,k])

    # Compute derivatives ∂/∂x, ∂/∂y, ∂/∂z of gamma_full
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

    # dgamma[..., dir, i, j] = ∂_dir γ_{ij}
    dgamma = np.zeros((Nx, Ny, Nz, 3, 3, 3))
    dgamma[..., 0, :, :] = dgamma_dx
    dgamma[..., 1, :, :] = dgamma_dy
    dgamma[..., 2, :, :] = dgamma_dz

    # Compute T[..., i, j, l] = ∂_i γ_{j l} + ∂_j γ_{i l} - ∂_l γ_{i j}
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

    # Gamma^i = gamma^{jk} Γ^i_{jk}
    Gamma = np.zeros((Nx, Ny, Nz, 3))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                Gamma[i,j,k] = np.sum(gamma_inv_full[i,j,k] * christoffels[i,j,k], axis=(0,1))

    return christoffels, Gamma

@jit(nopython=True)
def _compute_ricci_for_metric_jit(Nx, Ny, Nz, gamma_sym6, christoffels, dx, dy, dz):
    """JIT-compiled Ricci tensor computation for a given metric."""
    # R_{ij} = ∂_k Γ^k_{ij} - ∂_j Γ^k_{ik} + Γ^k_{ij} Γ^l_{kl} - Γ^k_{il} Γ^l_{kj}

    # Compute derivatives of Christoffels
    d_christ_dx = np.zeros((Nx, Ny, Nz, 3, 3, 3))
    d_christ_dy = np.zeros((Nx, Ny, Nz, 3, 3, 3))
    d_christ_dz = np.zeros((Nx, Ny, Nz, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                d_christ_dx[:, :, :, i, j, k] = np.gradient(christoffels[:, :, :, i, j, k], dx, axis=0)
                d_christ_dy[:, :, :, i, j, k] = np.gradient(christoffels[:, :, :, i, j, k], dy, axis=1)
                d_christ_dz[:, :, :, i, j, k] = np.gradient(christoffels[:, :, :, i, j, k], dz, axis=2)

    # ∂_k Γ^k_{ij}
    term1 = np.zeros((Nx, Ny, Nz, 3, 3))
    term1 += d_christ_dx.sum(axis=3)
    term1 += d_christ_dy.sum(axis=3)
    term1 += d_christ_dz.sum(axis=3)

    # - ∂_j Γ^k_{ik}
    term2 = np.zeros((Nx, Ny, Nz, 3, 3))
    for j in range(3):
        d_christ = [d_christ_dx, d_christ_dy, d_christ_dz][j]
        for i in range(3):
            term2[:, :, :, i, j] = -d_christ[:, :, :, np.arange(3), i, np.arange(3)].sum(axis=-1)

    # Γ^k_{ij} Γ^l_{kl}
    term3 = np.einsum('...kij,...lkl->...ij', christoffels, christoffels)

    # - Γ^k_{il} Γ^l_{kj}
    term4 = np.einsum('...kil,...lkj->...ij', christoffels, christoffels)

    ricci = term1 + term2 + term3 - term4
    return ricci

# Use the one from gr_core_fields

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
        gamma_hash = hash_array(self.fields.gamma_sym6)
        if gamma_hash in self._christoffel_cache:
            self.christoffels, self.Gamma = self._christoffel_cache[gamma_hash]
            self._christoffel_cache.move_to_end(gamma_hash)
            return

        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz
        gamma = self.fields.gamma_sym6  # (Nx, Ny, Nz, 6)

        self.christoffels, self.Gamma = compute_christoffels_compiled(gamma, dx, dy, dz)

        # Cache the result
        self._christoffel_cache[gamma_hash] = (self.christoffels.copy(), self.Gamma.copy())
        if len(self._christoffel_cache) > self._cache_maxsize:
            self._christoffel_cache.popitem(last=False)

    def compute_ricci_for_metric(self, gamma_sym6, christoffels):
        """Compute Ricci tensor R_{ij} for a given metric and its Christoffels using compiled function."""
        combined_hash = hash_array(np.concatenate([gamma_sym6.flatten(), christoffels.flatten()]))
        if combined_hash in self._ricci_for_metric_cache:
            return self._ricci_for_metric_cache[combined_hash]

        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz

        ricci = compute_ricci_compiled(gamma_sym6, christoffels, dx, dy, dz)

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

        # Compute \tilde{R}_{ij} using the conformal metric
        conformal_christoffels = np.zeros_like(self.christoffels)
        gamma_tilde = self.fields.gamma_tilde_sym6
        # Compute Christoffels for gamma_tilde
        self.compute_christoffels_for_metric(gamma_tilde, conformal_christoffels)
        R_tilde = self.compute_ricci_for_metric(gamma_tilde, conformal_christoffels)

        # Now, compute the phi terms
        chi = self.fields.phi
        if np.max(np.abs(chi)) < 1e-14:
            # If chi is zero, R_ij = R_tilde_ij
            self.ricci = R_tilde
        else:
            Nx, Ny, Nz = self.fields.Nx, self.fields.Ny, self.fields.Nz
            dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz

            # grad_chi
            grad_chi = np.zeros((Nx, Ny, Nz, 3))
            # The BSSN formula for R_ij requires conformal covariant derivatives of chi.
            # For a scalar, the conformal covariant derivative is the same as the partial derivative.
            grad_chi[..., 0] = np.gradient(chi, dx, axis=0) # \tilde{D}_x \chi = \partial_x \chi
            grad_chi[..., 1] = np.gradient(chi, dy, axis=1) # \tilde{D}_y \chi = \partial_y \chi
            grad_chi[..., 2] = np.gradient(chi, dz, axis=2) # \tilde{D}_z \chi = \partial_z \chi

            # DD_chi = D_i D_j chi
            # This should be the second *conformal* covariant derivative: \tilde{D}_i \tilde{D}_j \chi
            DD_chi = self.second_covariant_derivative_scalar(chi, christoffels=conformal_christoffels)

            # Lap_chi = gamma^{ij} D_i D_j chi
            # This should be the conformal Laplacian: \tilde{gamma}^{ij} \tilde{D}_i \tilde{D}_j \chi
            gamma_tilde_inv = inv_sym6(gamma_tilde)
            gamma_tilde_inv_full = sym6_to_mat33(gamma_tilde_inv)
            Lap_chi = np.einsum('...ij,...ij', gamma_tilde_inv_full, DD_chi)

            # D^k chi D_k chi
            # This should be with the conformal metric: \tilde{D}^k \chi \tilde{D}_k \chi
            D_chi_D_chi = np.einsum('...ij,...i,...j', gamma_tilde_inv_full, grad_chi, grad_chi)

            # The formula relates R_ij to \tilde{R}_ij. The extra terms involve chi and the conformal metric.
            # R_ij = \tilde{R}_ij - 2 \tilde{D}_i \tilde{D}_j \chi - 2 \tilde{\gamma}_{ij} \tilde{\Delta}\chi + 4 (\tilde{D}_i\chi)(\tilde{D}_j\chi) - 4 \tilde{\gamma}_{ij} (\tilde{D}_k\chi)(\tilde{D}^k\chi)
            # where \tilde{\Delta} is the conformal Laplacian.
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

        christoffels_out[...], _ = _compute_christoffels_jit(Nx, Ny, Nz, gamma_sym6, dx, dy, dz)

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

        # ∂_k V^i
        grad_V = np.zeros((Nx, Ny, Nz, 3, 3))  # grad_V[k, i] = ∂_k V^i
        for i in range(3):
            grad_V[..., 0, i] = np.gradient(V[..., i], dx, axis=0)
            grad_V[..., 1, i] = np.gradient(V[..., i], dy, axis=1)
            grad_V[..., 2, i] = np.gradient(V[..., i], dz, axis=2)

        # D_k V^i = ∂_k V^i + Γ^i_{jk} V^j
        D_V = grad_V + np.einsum('...ijk,...j->...ki', self.christoffels, V)

        # Cache the result
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
            # Use a hash of the provided christoffels as a proxy for the metric hash
            gamma_hash = hash_array(christoffels)
        combined_hash = gamma_hash + '_' + hash_array(scalar.flatten())
        if combined_hash in self._second_cov_deriv_scalar_cache:
            return self._second_cov_deriv_scalar_cache[combined_hash]

        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz

        DD_scalar = second_covariant_derivative_scalar_compiled(scalar, christoffels_to_use, dx, dy, dz)

        # Cache the result
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

        # Cache the result
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

        # Cache the result
        self._lie_K_cache[combined_hash] = lie_K_sym6.copy()
        if len(self._lie_K_cache) > self._cache_maxsize:
            self._lie_K_cache.popitem(last=False)

        return lie_K_sym6

    def compute_constraint_proxy(self, scale: str) -> float:
        """Compute proxy constraint residual for a given scale."""
        if not hasattr(self, 'R') or self.R is None:
            self.compute_scalar_curvature()

        if scale == 'L':
            # Large scale: mean absolute value
            return np.abs(self.R).mean()
        elif scale == 'M':
            # Medium scale: gradient magnitude
            return np.abs(np.gradient(self.R)).mean()
        else:  # 'H'
            # High scale: max absolute value
            return np.abs(self.R).max()
