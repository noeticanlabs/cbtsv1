# Implements: LoC-1 tangency/invariance via Bianchi identities for GR constraint propagation (H,M remain zero if initially zero under ADM/BSSN dynamics).

# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time"
]
LEXICON_SYMBOLS = {
    "\\mathcal{H}": "GR_constraint.hamiltonian",
    "\\mathcal{M}^i": "GR_constraint.momentum"
}

import numpy as np
import logging
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from .logging_config import array_stats
from .gr_core_fields import inv_sym6, trace_sym6, norm2_sym6, sym6_to_mat33, mat33_to_sym6
from gr_constraints_nsc import compute_hamiltonian_compiled, compute_momentum_compiled, discrete_L2_norm_compiled

logger = logging.getLogger('gr_solver.constraints')

def discrete_L2_norm(field, dx, dy, dz):
    """Compute discrete L2 norm: sqrt( sum(field^2) * dV )"""
    dV = dx * dy * dz
    return np.sqrt(np.sum(field**2) * dV)

def discrete_Linf_norm(field):
    """Compute discrete Linf norm: max(|field|)"""
    return np.max(np.abs(field))

class GRConstraints:
    def __init__(self, fields, geometry, num_workers: int = None):
        self.fields = fields
        self.geometry = geometry
        self.num_workers = num_workers or min(mp.cpu_count(), 8)
        self.executor = ProcessPoolExecutor(max_workers=self.num_workers)  # Reuse executor

    def _compute_hamiltonian_chunk(self, slice_indices, R, gamma_sym6, K_sym6):
        """Compute Hamiltonian for a chunk of the grid."""
        gamma_inv = inv_sym6(gamma_sym6[slice_indices])
        K_trace = trace_sym6(K_sym6[slice_indices], gamma_inv)
        K_sq = norm2_sym6(K_sym6[slice_indices], gamma_inv)
        return R[slice_indices] + K_trace**2 - K_sq

    def compute_hamiltonian(self):
        """Compute Hamiltonian constraint \mathcal{H} = R + K^2 - K_{ij}K^{ij} - 2\Lambda using compiled function."""
        start_time = time.time()

        # Ensure geometry R is up to date
        if not hasattr(self.geometry, 'R') or self.geometry.R is None:
            self.geometry.compute_scalar_curvature()

        self.H = compute_hamiltonian_compiled(self.geometry.R, self.fields.gamma_sym6, self.fields.K_sym6, self.fields.Lambda)

        elapsed = time.time() - start_time
        logger.info(f"Hamiltonian constraint computation time: {elapsed:.4f}s (compiled)")



    def compute_momentum(self):
        """Compute momentum constraints \mathcal{M}^i = D_j (K^{ij} - \gamma^{ij} K)."""
        start_time = time.time()
        Nx, Ny, Nz = self.fields.Nx, self.fields.Ny, self.fields.Nz
        self.M = np.zeros((Nx, Ny, Nz, 3))

        # Compute gamma_inv
        gamma_inv = inv_sym6(self.fields.gamma_sym6)
        gamma_inv_full = sym6_to_mat33(gamma_inv)

        # Compute K_trace = gamma^{ij} K_ij
        K_trace = trace_sym6(self.fields.K_sym6, gamma_inv)

        # Compute K^{ij} = gamma^{ik} gamma^{jl} K_kl
        K_full = sym6_to_mat33(self.fields.K_sym6)
        K_contravariant = np.einsum('...ik,...jl,...kl->...ij', gamma_inv_full, gamma_inv_full, K_full)

        # S^{ij} = K^{ij} - gamma^{ij} K
        gamma_full = sym6_to_mat33(self.fields.gamma_sym6)
        gamma_contravariant = gamma_inv_full  # since gamma^{ij} = inv(gamma)_{ij}
        K_expanded = K_trace[..., np.newaxis, np.newaxis] * gamma_contravariant
        S_ij = K_contravariant - K_expanded

        # Now, compute D_j S^{j i} = ∂_j S^{ji} + Γ^i_{jk} S^{kj} + Γ^j_{jk} S^{ki}
        # Note S is symmetric, so S^{ji} = S^{ij} etc.

        # Get Christoffels from geometry
        if not hasattr(self.geometry, 'christoffels') or self.geometry.christoffels is None:
            self.geometry.compute_christoffels()

        # First term: ∂_j S^{j i}
        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz
        grad_S_x = np.gradient(S_ij[..., 0, :], dx, axis=0)
        grad_S_y = np.gradient(S_ij[..., 1, :], dy, axis=1)
        grad_S_z = np.gradient(S_ij[..., 2, :], dz, axis=2)
        partial_div = grad_S_x + grad_S_y + grad_S_z

        # Second term: Γ^i_{jk} S^{jk}
        gamma_term1 = np.einsum('...ijk,...jk->...i', self.geometry.christoffels, S_ij)

        # Third term: Γ^j_{jk} S^{ki}
        # where Γ^j_{jk} = ∂_k ln(sqrt(det(γ)))
        from .gr_core_fields import det_sym6
        det_gamma = det_sym6(self.fields.gamma_sym6)
        log_sqrt_det_gamma = 0.5 * np.log(det_gamma)
        grad_log_sqrt_det_gamma = np.stack(np.gradient(log_sqrt_det_gamma, dx, dy, dz), axis=-1)
        gamma_term2 = np.einsum('...k,...ki->...i', grad_log_sqrt_det_gamma, S_ij)

        self.M = partial_div + gamma_term1 + gamma_term2

        elapsed = time.time() - start_time
        logger.info(f"Momentum constraint computation time: {elapsed:.4f}s (sequential)")

    def compute_residuals(self):
        """Compute residuals: eps_H (L2 H), eps_M (L2 sqrt(gamma^{ij} M_i M_j)), eps_proj (L2 aux constraints)."""
        # L2 norm: sqrt( sum field^2 dV ) using compiled function
        self.eps_H = discrete_L2_norm_compiled(self.H, self.fields.dx, self.fields.dy, self.fields.dz)
        self.eps_H_grid = self.H  # Grid of Hamiltonian constraint values

        # Compute invariant M_norm = sqrt( gamma^{ij} M_i M_j )
        gamma_inv = inv_sym6(self.fields.gamma_sym6)
        gamma_inv_full = sym6_to_mat33(gamma_inv)
        M_norm_squared = np.sum(gamma_inv_full * self.M[..., np.newaxis, :] * self.M[..., :, np.newaxis], axis=(-2, -1))
        M_norm = np.sqrt(M_norm_squared)
        self.eps_M = discrete_L2_norm_compiled(M_norm, self.fields.dx, self.fields.dy, self.fields.dz)

        # eps_proj: L2 of auxiliary constraints Z and Z_i (BSSN-Z4)
        # Assuming Z and Z_i are available in fields
        if hasattr(self.fields, 'Z') and hasattr(self.fields, 'Z_i'):
            Z_combined = np.concatenate([self.fields.Z[..., np.newaxis], self.fields.Z_i], axis=-1)
            self.eps_proj = discrete_L2_norm(Z_combined, self.fields.dx, self.fields.dy, self.fields.dz)
        else:
            self.eps_proj = 0.0

        # eps_clk will be computed in stepper, as it requires stage information
        self.eps_clk = None  # Placeholder

        # Normalized versions (divide by initial if available, else 1)
        self.eps_H_norm = self.eps_H / max(1e-20, getattr(self, 'eps_H_initial', 1.0))
        self.eps_M_norm = self.eps_M / max(1e-20, getattr(self, 'eps_M_initial', 1.0))
        self.eps_proj_norm = self.eps_proj / max(1e-20, getattr(self, 'eps_proj_initial', 1.0))

        logger.debug("Computed constraint residuals", extra={
            "extra_data": {
                "eps_H": float(self.eps_H),
                "eps_M": float(self.eps_M),
                "eps_proj": float(self.eps_proj),
                "eps_H_norm": float(self.eps_H_norm),
                "eps_M_norm": float(self.eps_M_norm),
                "eps_proj_norm": float(self.eps_proj_norm),
                "H_stats": array_stats(self.H, "H"),
                "M_stats": array_stats(self.M, "M")
            }
        })

    def compute_stage_difference_Linf(self, Psi_used, Psi_auth):
        """Compute Linf norm of difference between Psi_used and Psi_auth states."""
        diffs = []
        for key in Psi_used:
            if key in Psi_auth:
                diff = np.max(np.abs(Psi_used[key] - Psi_auth[key]))
                diffs.append(diff)
        return max(diffs) if diffs else 0.0

    @staticmethod
    def compute_scaling_law(eps1, eps2, h1, h2):
        """Compute observed convergence order p_obs = log(eps1/eps2) / log(h1/h2)."""
        if eps1 <= 0 or eps2 <= 0 or h1 <= 0 or h2 <= 0:
            return float('nan')
        ratio_eps = eps1 / eps2
        ratio_h = h1 / h2
        if ratio_eps <= 0 or ratio_h <= 0 or ratio_h == 1:
            return float('nan')
        return np.log(ratio_eps) / np.log(ratio_h)