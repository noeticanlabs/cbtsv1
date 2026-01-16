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
from .gr_core_fields import inv_sym6, sym6_to_mat33, mat33_to_sym6

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

    def compute_christoffels(self):
        """Compute Christoffel symbols \\Gamma^k_{ij} and Gamma^i using finite differences."""
        Nx, Ny, Nz = self.fields.Nx, self.fields.Ny, self.fields.Nz
        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz
        gamma = self.fields.gamma_sym6  # (Nx, Ny, Nz, 6)

        # Convert to full 3x3 tensor for derivatives
        gamma_full = sym6_to_mat33(gamma)

        # Compute derivatives ∂/∂x, ∂/∂y, ∂/∂z of gamma_full
        dgamma_dx = np.gradient(gamma_full, dx, axis=0)
        dgamma_dy = np.gradient(gamma_full, dy, axis=1)
        dgamma_dz = np.gradient(gamma_full, dz, axis=2)

        # Christoffels \\Gamma^k_{ij} = 1/2 gamma^{kl} (∂_i gamma_{jl} + ∂_j gamma_{il} - ∂_l gamma_{ij})
        from .gr_core_fields import inv_sym6
        gamma_inv = inv_sym6(gamma)  # (Nx,Ny,Nz,6)

        gamma_inv_full = sym6_to_mat33(gamma_inv)

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

        self.christoffels = 0.5 * np.einsum('...kl,...ijl->...kij', gamma_inv_full, T)

        # Gamma^i = gamma^{jk} \\Gamma^i_{jk}
        self.Gamma = np.einsum('...jk,...ijk->...i', gamma_inv_full, self.christoffels)

    def compute_ricci_for_metric(self, gamma_sym6, christoffels):
        """Compute Ricci tensor R_{ij} for a given metric and its Christoffels."""
        Nx, Ny, Nz = self.fields.Nx, self.fields.Ny, self.fields.Nz
        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz

        # R_{ij} = ∂_k Γ^k_{ij} - ∂_j Γ^k_{ik} + Γ^k_{ij} Γ^l_{kl} - Γ^k_{il} Γ^l_{kj}

        # Compute derivatives of Christoffels
        d_christ_dx = np.gradient(christoffels, dx, axis=0)
        d_christ_dy = np.gradient(christoffels, dy, axis=1)
        d_christ_dz = np.gradient(christoffels, dz, axis=2)

        # ∂_k Γ^k_{ij}
        term1 = np.sum(d_christ_dx, axis=3) + np.sum(d_christ_dy, axis=3) + np.sum(d_christ_dz, axis=3)

        # - ∂_j Γ^k_{ik}
        term2 = np.zeros((Nx, Ny, Nz, 3, 3))
        for j in range(3):
            d_christ = [d_christ_dx, d_christ_dy, d_christ_dz][j]
            for i in range(3):
                term2[..., i, j] = -np.sum(d_christ[..., np.arange(3), i, np.arange(3)], axis=-1)

        # Γ^k_{ij} Γ^l_{kl}
        term3 = np.einsum('...kij,...lkl->...ij', christoffels, christoffels)

        # - Γ^k_{il} Γ^l_{kj}
        term4 = np.einsum('...kil,...lkj->...ij', christoffels, christoffels)

        ricci = term1 + term2 + term3 - term4

        return ricci

    def compute_ricci(self):
        """Compute Ricci tensor R_{ij} using BSSN conformal decomposition."""
        # Compute \tilde{R}_{ij} using the conformal metric
        conformal_christoffels = np.zeros_like(self.christoffels)
        gamma_tilde = self.fields.gamma_tilde_sym6
        # Compute Christoffels for gamma_tilde
        self.compute_christoffels_for_metric(gamma_tilde, conformal_christoffels)
        R_tilde = self.compute_ricci_for_metric(gamma_tilde, conformal_christoffels)

        # Now, compute the phi terms
        phi = self.fields.phi
        if np.max(np.abs(phi)) < 1e-14:
            # If phi is zero, R_ij = R_tilde_ij
            self.ricci = R_tilde
            return

        # Compute physical Christoffels
        self.compute_christoffels()

        Nx, Ny, Nz = self.fields.Nx, self.fields.Ny, self.fields.Nz
        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz

        # grad_phi
        grad_phi = np.zeros((Nx, Ny, Nz, 3))
        grad_phi[..., 0] = np.gradient(phi, dx, axis=0)
        grad_phi[..., 1] = np.gradient(phi, dy, axis=1)
        grad_phi[..., 2] = np.gradient(phi, dz, axis=2)

        # DD_phi = D_i D_j phi
        DD_phi = self.second_covariant_derivative_scalar(phi)

        # Lap_phi = gamma^{ij} D_i D_j phi
        gamma_inv = inv_sym6(self.fields.gamma_sym6)
        gamma_inv_full = sym6_to_mat33(gamma_inv)
        Lap_phi = np.einsum('...ij,...ij', gamma_inv_full, DD_phi)

        # D^k phi D_k phi
        D_phi_D_phi = np.einsum('...ij,...i,...j', gamma_inv_full, grad_phi, grad_phi)

        gamma_full = sym6_to_mat33(self.fields.gamma_sym6)

        R_phi = -2 * DD_phi - 2 * gamma_full * Lap_phi[..., np.newaxis, np.newaxis] \
                + 4 * np.einsum('...i,...j->...ij', grad_phi, grad_phi) \
                - 4 * gamma_full * D_phi_D_phi[..., np.newaxis, np.newaxis]

        self.ricci = R_tilde + R_phi

    def compute_christoffels_for_metric(self, gamma_sym6, christoffels_out):
        """Compute Christoffel symbols for a given metric."""
        Nx, Ny, Nz = self.fields.Nx, self.fields.Ny, self.fields.Nz
        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz

        gamma_full = sym6_to_mat33(gamma_sym6)
        dgamma_dx = np.gradient(gamma_full, dx, axis=0)
        dgamma_dy = np.gradient(gamma_full, dy, axis=1)
        dgamma_dz = np.gradient(gamma_full, dz, axis=2)

        gamma_inv = inv_sym6(gamma_sym6)
        gamma_inv_full = sym6_to_mat33(gamma_inv)

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

        christoffels_out[...] = 0.5 * np.einsum('...kl,...ijl->...kij', gamma_inv_full, T)

    def compute_scalar_curvature(self):
        """Compute scalar curvature R = gamma^{ij} R_{ij}."""
        if not hasattr(self, 'ricci') or self.ricci is None:
            self.compute_ricci()

        gamma_inv = inv_sym6(self.fields.gamma_sym6)
        gamma_inv_full = sym6_to_mat33(gamma_inv)

        self.R = np.einsum('...ij,...ij', gamma_inv_full, self.ricci)

    def covariant_derivative_vector(self, V):
        """Compute covariant derivative D_k V^i = ∂_k V^i + Γ^i_{jk} V^j"""
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

        return D_V

    def second_covariant_derivative_scalar(self, scalar):
        """Compute D_i D_j scalar = ∂_i ∂_j scalar - Γ^k_{ij} ∂_k scalar"""
        if not hasattr(self, 'christoffels') or self.christoffels is None:
            self.compute_christoffels()

        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz
        Nx, Ny, Nz = self.fields.Nx, self.fields.Ny, self.fields.Nz

        # ∂_k scalar
        grad_scalar = np.zeros((Nx, Ny, Nz, 3))
        grad_scalar[..., 0] = np.gradient(scalar, dx, axis=0)
        grad_scalar[..., 1] = np.gradient(scalar, dy, axis=1)
        grad_scalar[..., 2] = np.gradient(scalar, dz, axis=2)

        # ∂_i ∂_j scalar
        hess_scalar = np.zeros((Nx, Ny, Nz, 3, 3))
        for i in range(3):
            hess_scalar[..., i, 0] = np.gradient(grad_scalar[..., i], dx, axis=0)
            hess_scalar[..., i, 1] = np.gradient(grad_scalar[..., i], dy, axis=1)
            hess_scalar[..., i, 2] = np.gradient(grad_scalar[..., i], dz, axis=2)

        # D_i D_j scalar = ∂_i ∂_j scalar - Γ^k_{ij} ∂_k scalar
        DD_scalar = hess_scalar - np.einsum('...kij,...k->...ij', self.christoffels, grad_scalar)

        return DD_scalar

    def lie_derivative_gamma(self, gamma_sym6, beta):
        """Compute Lie derivative L_β γ_ij = β^k ∂_k γ_ij + γ_kj ∂_i β^k + γ_ik ∂_j β^k"""
        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz
        Nx, Ny, Nz = self.fields.Nx, self.fields.Ny, self.fields.Nz

        # Convert to full tensor
        gamma_full = sym6_to_mat33(gamma_sym6)

        # ∂_k γ_ij
        dgamma_dx = np.gradient(gamma_full, dx, axis=0)
        dgamma_dy = np.gradient(gamma_full, dy, axis=1)
        dgamma_dz = np.gradient(gamma_full, dz, axis=2)
        dgamma_d = np.stack([dgamma_dx, dgamma_dy, dgamma_dz], axis=-4)  # (3, Nx, Ny, Nz, 3, 3)

        # β^k ∂_k γ_ij
        lie_term1 = np.einsum('...k,k...ij->...ij', beta, dgamma_d)

        # γ_kj ∂_i β^k + γ_ik ∂_j β^k
        grad_beta = np.zeros((Nx, Ny, Nz, 3, 3))
        for k in range(3):
            grad_beta[..., k, 0] = np.gradient(beta[..., k], dx, axis=0)
            grad_beta[..., k, 1] = np.gradient(beta[..., k], dy, axis=1)
            grad_beta[..., k, 2] = np.gradient(beta[..., k], dz, axis=2)

        lie_term2 = np.einsum('...kj,...ki->...ij', gamma_full, grad_beta) + np.einsum('...ik,...kj->...ij', gamma_full, grad_beta)

        lie_gamma_full = lie_term1 + lie_term2
        lie_gamma_sym6 = mat33_to_sym6(lie_gamma_full)
        return lie_gamma_sym6

    def lie_derivative_K(self, K_sym6, beta):
        """Compute Lie derivative L_β K_ij = β^k ∂_k K_ij + K_kj ∂_i β^k + K_ik ∂_j β^k"""
        # Similar to gamma, but for K_sym6
        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz
        Nx, Ny, Nz = self.fields.Nx, self.fields.Ny, self.fields.Nz

        # Convert to full tensor
        K_full = sym6_to_mat33(K_sym6)

        # ∂_k K_ij
        dK_dx = np.gradient(K_full, dx, axis=0)
        dK_dy = np.gradient(K_full, dy, axis=1)
        dK_dz = np.gradient(K_full, dz, axis=2)
        dK_d = np.stack([dK_dx, dK_dy, dK_dz], axis=-4)  # (3, Nx, Ny, Nz, 3, 3)

        # β^k ∂_k K_ij
        lie_term1 = np.einsum('...k,k...ij->...ij', beta, dK_d)

        # K_kj ∂_i β^k + K_ik ∂_j β^k
        grad_beta = np.zeros((Nx, Ny, Nz, 3, 3))
        for k in range(3):
            grad_beta[..., k, 0] = np.gradient(beta[..., k], dx, axis=0)
            grad_beta[..., k, 1] = np.gradient(beta[..., k], dy, axis=1)
            grad_beta[..., k, 2] = np.gradient(beta[..., k], dz, axis=2)

        lie_term2 = np.einsum('...kj,...ki->...ij', K_full, grad_beta) + np.einsum('...ik,...kj->...ij', K_full, grad_beta)

        lie_K_full = lie_term1 + lie_term2
        lie_K_sym6 = mat33_to_sym6(lie_K_full)
        return lie_K_sym6
