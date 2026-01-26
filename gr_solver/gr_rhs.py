import numpy as np
import logging
try:
    from numba import jit, prange
except ImportError:
    jit = lambda f=None, **kwargs: f if f else (lambda g: g)
    prange = range
from .logging_config import Timer
from .gr_core_fields import inv_sym6, trace_sym6, sym6_to_mat33, mat33_to_sym6, det_sym6
from .gr_geometry_nsc import _sym6_to_mat33_jit, _inv_sym6_jit

logger = logging.getLogger('gr_solver.rhs')

@jit(nopython=True)
def _compute_gamma_tilde_rhs_jit(Nx, Ny, Nz, alpha, beta, phi, Gamma_tilde, A_sym6, gamma_tilde_sym6, dx, dy, dz, K_trace_scratch):
    """JIT-compiled computation of Gamma_tilde RHS."""
    raise NotImplementedError("This function is a performance bottleneck and has been replaced. Use the algebraic JIT version.")

@jit(nopython=True, parallel=True, fastmath=True)
def _calculate_rhs_Gamma_tilde_algebraic_jit(
    alpha, beta, Gamma_tilde,
    gamma_tilde_inv_full, A_tilde_uu, christoffel_tilde_udd,
    dalpha, dphi, dK, dGamma, dbeta, lap_beta, d_div_beta, div_beta
):
    Nx, Ny, Nz = alpha.shape
    rhs = np.zeros((Nx, Ny, Nz, 3))

    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # A: advection: beta^k * d_k Gamma_tilde^i
                advection_term = np.zeros(3)
                for i_comp in range(3):
                    for k_comp in range(3):
                        advection_term[i_comp] += beta[i,j,k,k_comp] * dGamma[i,j,k,i_comp,k_comp]

                # B: stretching: -Gamma_tilde^k * d_k beta^i + 2/3 Gamma_tilde^i * d_k beta^k
                stretching_term = np.zeros(3)
                for i_comp in range(3):
                    term1 = 0.0
                    for k_comp in range(3):
                        term1 += Gamma_tilde[i,j,k,k_comp] * dbeta[i,j,k,k_comp,i_comp]
                    stretching_term[i_comp] = -term1 + (2.0/3.0) * Gamma_tilde[i,j,k,i_comp] * div_beta[i,j,k]
                
                # C: shift second-derivatives: lap_beta^i + 1/3 gamma_tilde^{ij} d_j(d_k beta^k)
                shift_deriv_term = np.zeros(3)
                for i_comp in range(3):
                    term2 = 0.0
                    for j_comp in range(3):
                        term2 += gamma_tilde_inv_full[i,j,k,i_comp,j_comp] * d_div_beta[i,j,k,j_comp]
                    shift_deriv_term[i_comp] = lap_beta[i,j,k,i_comp] + (1.0/3.0) * term2

                # D: lapse/curvature
                # -2 A_tilde^{ij} d_j alpha
                term_d_alpha = np.zeros(3)
                for i_comp in range(3):
                    for j_comp in range(3):
                        term_d_alpha[i_comp] -= 2.0 * A_tilde_uu[i,j,k,i_comp,j_comp] * dalpha[i,j,k,j_comp]

                # 2 alpha (Gamma_tilde^i_{jk} A_tilde^{jk} + 6 A_tilde^{ij} d_j phi - 2/3 gamma_tilde^{ij} d_j K)
                GammaA = np.zeros(3)
                for i_comp in range(3):
                    for j_comp in range(3):
                        for k_comp in range(3):
                            GammaA[i_comp] += christoffel_tilde_udd[i,j,k,i_comp,j_comp,k_comp] * A_tilde_uu[i,j,k,j_comp,k_comp]
                
                A_dphi = np.zeros(3)
                for i_comp in range(3):
                    for j_comp in range(3):
                        A_dphi[i_comp] += A_tilde_uu[i,j,k,i_comp,j_comp] * dphi[i,j,k,j_comp]

                gamma_dK = np.zeros(3)
                for i_comp in range(3):
                    for j_comp in range(3):
                        gamma_dK[i_comp] += gamma_tilde_inv_full[i,j,k,i_comp,j_comp] * dK[i,j,k,j_comp]
                
                lapse_curv_term = np.zeros(3)
                for i_comp in range(3):
                    lapse_curv_term[i_comp] = 2.0 * alpha[i,j,k] * (GammaA[i_comp] + 6.0 * A_dphi[i_comp] - (2.0/3.0) * gamma_dK[i_comp])

                for i_comp in range(3):
                    rhs[i,j,k,i_comp] = advection_term[i_comp] + stretching_term[i_comp] + shift_deriv_term[i_comp] + term_d_alpha[i_comp] + lapse_curv_term[i_comp]
    
    return rhs

class GRRhs:
    def __init__(self, fields, geometry, constraints, loc_operator=None, lambda_val=0.0, sources_func=None, aeonic_mode=True):
        self.fields = fields
        self.geometry = geometry
        self.constraints = constraints
        self.loc_operator = loc_operator
        self.lambda_val = lambda_val
        self.sources_func = sources_func
        self.aeonic_mode = aeonic_mode
        
        # Load Hadamard RHS if available
        # Disabling HadamardVM path due to performance issues with monolithic JIT kernels.
        self.rhs_func = None

        if self.aeonic_mode:
            # Preallocate RHS arrays
            self.rhs_gamma_sym6 = np.zeros_like(self.fields.gamma_sym6)
            self.rhs_K_sym6 = np.zeros_like(self.fields.K_sym6)
            self.rhs_phi = np.zeros_like(self.fields.phi)
            self.rhs_gamma_tilde_sym6 = np.zeros_like(self.fields.gamma_tilde_sym6)
            self.rhs_A_sym6 = np.zeros_like(self.fields.A_sym6)
            self.rhs_Gamma_tilde = np.zeros_like(self.fields.Gamma_tilde)
            self.rhs_Z = np.zeros_like(self.fields.Z)
            self.rhs_Z_i = np.zeros_like(self.fields.Z_i)

            # Sources arrays
            self.S_gamma_sym6 = np.zeros_like(self.fields.gamma_sym6)
            self.S_K_sym6 = np.zeros_like(self.fields.K_sym6)
            self.S_gamma_tilde_sym6 = np.zeros_like(self.fields.gamma_tilde_sym6)
            self.S_A_sym6 = np.zeros_like(self.fields.A_sym6)
            self.S_phi = np.zeros_like(self.fields.phi)
            self.S_Gamma_tilde = np.zeros_like(self.fields.Gamma_tilde)
            self.S_Z = np.zeros_like(self.fields.Z)
            self.S_Z_i = np.zeros_like(self.fields.Z_i)

            # Preallocate scratch buffers
            self.gamma_inv_scratch = np.zeros_like(self.fields.gamma_sym6)
            self.K_trace_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz))
            self.alpha_expanded_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 6))
            self.alpha_expanded_33_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
            self.lie_gamma_scratch = np.zeros_like(self.fields.gamma_sym6)
            self.DD_alpha_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
            self.DD_alpha_sym6_scratch = np.zeros_like(self.fields.gamma_sym6)
            self.ricci_sym6_scratch = np.zeros_like(self.fields.gamma_sym6)
            self.K_full_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
            self.gamma_inv_full_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
            self.K_contracted_full_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
            self.K_contracted_sym6_scratch = np.zeros_like(self.fields.gamma_sym6)
            self.lie_K_scratch = np.zeros_like(self.fields.gamma_sym6)
            self.lie_gamma_tilde_scratch = np.zeros_like(self.fields.gamma_tilde_sym6)
            self.psi_minus4_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz))
            self.psi_minus4_expanded_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 6))
            self.ricci_tf_sym6_scratch = np.zeros_like(self.fields.gamma_sym6)
            self.rhs_A_temp_scratch = np.zeros_like(self.fields.gamma_sym6)
            self.lie_A_scratch = np.zeros_like(self.fields.gamma_sym6)

    def compute_rhs(self, t=0.0, slow_update=True):
        """Compute full ADM RHS B with spatial derivatives."""
        rhs_timer = Timer("compute_rhs")
        with rhs_timer:
            if self.rhs_func:
                # Use Hadamard VM via rhs_func
                fields_dict = {
                    'gamma_sym6': self.fields.gamma_sym6,
                    'K_sym6': self.fields.K_sym6,
                    'alpha': self.fields.alpha,
                    'beta': self.fields.beta,
                    'phi': self.fields.phi,
                    'gamma_tilde_sym6': self.fields.gamma_tilde_sym6,
                    'A_sym6': self.fields.A_sym6,
                    'Gamma_tilde': self.fields.Gamma_tilde,
                    'Z': self.fields.Z,
                    'Z_i': self.fields.Z_i,
                    'dx': self.fields.dx, 'dy': self.fields.dy, 'dz': self.fields.dz
                }
                rhs_result = self.rhs_func(fields_dict, lambda_val=self.lambda_val, sources_enabled=self.sources_func is not None)
                # Map to self.rhs_*
                self.rhs_gamma_sym6[:] = rhs_result['rhs_gamma_sym6']
                self.rhs_K_sym6[:] = rhs_result['rhs_K_sym6']
                self.rhs_phi[:] = rhs_result['rhs_phi']
                self.rhs_gamma_tilde_sym6[:] = rhs_result['rhs_gamma_tilde_sym6']
                self.rhs_A_sym6[:] = rhs_result['rhs_A_sym6']
                self.rhs_Gamma_tilde[:] = rhs_result['rhs_Gamma_tilde']
                self.rhs_Z[:] = rhs_result['rhs_Z']
                self.rhs_Z_i[:] = rhs_result['rhs_Z_i']
                return

        if self.aeonic_mode:
            # Use preallocated arrays
            pass # Variables already set in self
        else:
            # Allocate on the fly (fallback)
            self.rhs_gamma_sym6 = np.zeros_like(self.fields.gamma_sym6)
            self.rhs_K_sym6 = np.zeros_like(self.fields.K_sym6)
            self.rhs_phi = np.zeros_like(self.fields.phi)
            self.rhs_gamma_tilde_sym6 = np.zeros_like(self.fields.gamma_tilde_sym6)
            self.rhs_A_sym6 = np.zeros_like(self.fields.A_sym6)
            self.rhs_Gamma_tilde = np.zeros_like(self.fields.Gamma_tilde)
            self.rhs_Z = np.zeros_like(self.fields.Z)
            self.rhs_Z_i = np.zeros_like(self.fields.Z_i)
            # ... (scratch buffers would also need allocation if not preallocated, but for brevity we assume aeonic_mode=True is standard)

        # Set sources if func provided
        if self.sources_func is not None:
            sources = self.sources_func(t)
            self.S_gamma_sym6[:] = sources.get('S_gamma_sym6', 0.0)
            self.S_K_sym6[:] = sources.get('S_K_sym6', 0.0)
            self.S_gamma_tilde_sym6[:] = sources['S_gamma_tilde_sym6']
            self.S_A_sym6[:] = sources['S_A_sym6']
            self.S_phi[:] = sources['S_phi']
            self.S_Gamma_tilde[:] = sources['S_Gamma_tilde']
            self.S_Z[:] = sources['S_Z']
            self.S_Z_i[:] = sources['S_Z_i']

        # Ensure geometry is computed
        if not hasattr(self.geometry, 'ricci') or self.geometry.ricci is None:
            self.geometry.compute_ricci()

        # Compute K = gamma^{ij} K_ij
        self.gamma_inv_scratch[:] = inv_sym6(self.fields.gamma_sym6)
        self.K_trace_scratch[:] = trace_sym6(self.fields.K_sym6, self.gamma_inv_scratch)

        self.alpha_expanded_scratch[:] = self.fields.alpha[..., np.newaxis]
        self.alpha_expanded_33_scratch[:] = self.fields.alpha[..., np.newaxis, np.newaxis]

        # ADM ∂t gamma_ij = -2 α K_ij + L_β γ_ij
        self.lie_gamma_scratch[:] = self.geometry.lie_derivative_gamma(self.fields.gamma_sym6, self.fields.beta)
        self.rhs_gamma_sym6[:] = -2.0 * self.alpha_expanded_scratch * self.fields.K_sym6 + self.lie_gamma_scratch
        self.rhs_gamma_sym6 += self.S_gamma_sym6

        # ADM ∂t K_ij
        self.DD_alpha_scratch[:] = self.geometry.second_covariant_derivative_scalar(self.fields.alpha)
        self.DD_alpha_sym6_scratch[:] = mat33_to_sym6(self.DD_alpha_scratch)
        self.ricci_sym6_scratch[:] = mat33_to_sym6(self.geometry.ricci)
        self.K_full_scratch[:] = sym6_to_mat33(self.fields.K_sym6)
        self.gamma_inv_full_scratch[:] = sym6_to_mat33(self.gamma_inv_scratch)
        self.K_contracted_full_scratch[:] = np.einsum('...ij,...jk,...kl->...il', self.K_full_scratch, self.gamma_inv_full_scratch, self.K_full_scratch)
        self.K_contracted_sym6_scratch[:] = mat33_to_sym6(self.K_contracted_full_scratch)
        self.lie_K_scratch[:] = self.geometry.lie_derivative_K(self.fields.K_sym6, self.fields.beta)
        lambda_term = 2.0 * self.alpha_expanded_scratch * self.fields.Lambda * self.fields.gamma_sym6

        self.rhs_K_sym6[:] = (-self.DD_alpha_sym6_scratch +
                              self.alpha_expanded_scratch * self.ricci_sym6_scratch +
                              -2.0 * self.alpha_expanded_scratch * self.K_contracted_sym6_scratch +
                              self.alpha_expanded_scratch * self.K_trace_scratch[..., np.newaxis] * self.fields.K_sym6 +
                              lambda_term +
                              self.lie_K_scratch)
        self.rhs_K_sym6 += self.S_K_sym6

        # BSSN ∂_0 φ
        if slow_update:
            div_beta = np.gradient(self.fields.beta[..., 0], self.fields.dx, axis=0) + \
                       np.gradient(self.fields.beta[..., 1], self.fields.dy, axis=1) + \
                       np.gradient(self.fields.beta[..., 2], self.fields.dz, axis=2)
            self.rhs_phi[:] = - (self.fields.alpha / 6.0) * self.K_trace_scratch + (1.0 / 6.0) * div_beta
            grad_phi = np.array([np.gradient(self.fields.phi, self.fields.dx, axis=0),
                                 np.gradient(self.fields.phi, self.fields.dy, axis=1),
                                 np.gradient(self.fields.phi, self.fields.dz, axis=2)])
            advection_phi = np.sum(self.fields.beta * grad_phi.transpose(1,2,3,0), axis=-1)
            self.rhs_phi[:] += advection_phi
            self.rhs_phi += self.S_phi
        else:
            self.rhs_phi[:] = 0.0

        # Full BSSN Gamma_tilde evolution
        # --- Prepare arguments for the algebraic JIT kernel ---
        Nx, Ny, Nz = self.fields.Nx, self.fields.Ny, self.fields.Nz
        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz

        # Precompute algebraic quantities
        gamma_tilde_inv = inv_sym6(self.fields.gamma_tilde_sym6)
        gamma_tilde_inv_full = sym6_to_mat33(gamma_tilde_inv)
        A_full = sym6_to_mat33(self.fields.A_sym6)
        A_tilde_uu = np.einsum('...ik,...kl,...lj->...ij', gamma_tilde_inv_full, A_full, gamma_tilde_inv_full)

        # Precompute Christoffel symbols for the conformal metric
        christoffel_tilde_udd = np.zeros_like(self.geometry.christoffels)
        self.geometry.compute_christoffels_for_metric(self.fields.gamma_tilde_sym6, christoffel_tilde_udd)

        # Precompute all required gradients
        dalpha = np.stack(np.gradient(self.fields.alpha, dx, dy, dz), axis=-1)
        dphi = np.stack(np.gradient(self.fields.phi, dx, dy, dz), axis=-1)

        dbeta = np.zeros((Nx, Ny, Nz, 3, 3))
        for k_idx in range(3):
            dbeta[..., k_idx, 0] = np.gradient(self.fields.beta[..., k_idx], dx, axis=0)
            dbeta[..., k_idx, 1] = np.gradient(self.fields.beta[..., k_idx], dy, axis=1)
            dbeta[..., k_idx, 2] = np.gradient(self.fields.beta[..., k_idx], dz, axis=2)

        lap_beta = np.zeros((Nx, Ny, Nz, 3))
        for i_idx in range(3):
            fxx = np.gradient(np.gradient(self.fields.beta[..., i_idx], dx, axis=0), dx, axis=0)
            fyy = np.gradient(np.gradient(self.fields.beta[..., i_idx], dy, axis=1), dy, axis=1)
            fzz = np.gradient(np.gradient(self.fields.beta[..., i_idx], dz, axis=2), dz, axis=2)
            lap_beta[..., i_idx] = fxx + fyy + fzz

        div_beta = np.gradient(self.fields.beta[..., 0], dx, axis=0) + \
                   np.gradient(self.fields.beta[..., 1], dy, axis=1) + \
                   np.gradient(self.fields.beta[..., 2], dz, axis=2)

        d_div_beta = np.stack(np.gradient(div_beta, dx, dy, dz), axis=-1)

        dGamma = np.zeros((Nx, Ny, Nz, 3, 3))
        for i_idx in range(3):
            dGamma[..., i_idx, 0] = np.gradient(self.fields.Gamma_tilde[..., i_idx], dx, axis=0)
            dGamma[..., i_idx, 1] = np.gradient(self.fields.Gamma_tilde[..., i_idx], dy, axis=1)
            dGamma[..., i_idx, 2] = np.gradient(self.fields.Gamma_tilde[..., i_idx], dz, axis=2)

        dK = np.stack(np.gradient(self.K_trace_scratch, dx, dy, dz), axis=-1)

        # Call the fast algebraic JIT kernel
        self.rhs_Gamma_tilde[:] = _calculate_rhs_Gamma_tilde_algebraic_jit(
            self.fields.alpha, self.fields.beta, self.fields.Gamma_tilde,
            gamma_tilde_inv_full, A_tilde_uu, christoffel_tilde_udd,
            dalpha, dphi, dK, dGamma, dbeta, lap_beta, d_div_beta, div_beta
        )

        # BSSN ∂t γ̃_ij
        self.lie_gamma_tilde_scratch[:] = self.geometry.lie_derivative_gamma(self.fields.gamma_tilde_sym6, self.fields.beta)
        self.rhs_gamma_tilde_sym6[:] = -2.0 * self.alpha_expanded_scratch * self.fields.A_sym6 + self.lie_gamma_tilde_scratch

        # BSSN ∂_0 A_ij
        self.psi_minus4_scratch[:] = np.exp(-4 * self.fields.phi)
        self.psi_minus4_expanded_scratch[:] = self.psi_minus4_scratch[..., np.newaxis]
        self.ricci_tf_sym6_scratch[:] = self.ricci_sym6_scratch - (1/3) * self.fields.gamma_sym6 * (self.geometry.R[..., np.newaxis] if hasattr(self.geometry, 'R') else 0)
        self.rhs_A_temp_scratch[:] = self.psi_minus4_expanded_scratch * self.alpha_expanded_scratch * self.ricci_tf_sym6_scratch + self.alpha_expanded_scratch * self.K_trace_scratch[..., np.newaxis] * self.fields.A_sym6
        self.lie_A_scratch[:] = self.geometry.lie_derivative_K(self.fields.A_sym6, self.fields.beta)
        self.rhs_A_sym6[:] = self.rhs_A_temp_scratch + self.lie_A_scratch

        # RHS for constraint damping
        if slow_update:
            self.constraints.compute_hamiltonian()
            kappa = 1.0
            self.rhs_Z[:] = -kappa * self.fields.alpha * self.constraints.H
            self.constraints.compute_momentum()
            self.rhs_Z_i[:] = -kappa * self.fields.alpha[:, :, :, np.newaxis] * self.constraints.M
            self.rhs_Z += self.S_Z
            self.rhs_Z_i += self.S_Z_i
        else:
            self.rhs_Z[:] = 0.0
            self.rhs_Z_i[:] = 0.0

        # Add sources
        self.rhs_gamma_tilde_sym6 += self.S_gamma_tilde_sym6
        self.rhs_A_sym6 += self.S_A_sym6
        self.rhs_Gamma_tilde += self.S_Gamma_tilde

        # Add LoC augmentation
        if self.lambda_val > 0 and self.loc_operator:
            K_LoC_scaled = self.loc_operator.get_K_LoC_for_rhs()
            self.rhs_gamma_sym6 += K_LoC_scaled['gamma_sym6']
            self.rhs_K_sym6 += K_LoC_scaled['K_sym6']
            self.rhs_gamma_tilde_sym6 += K_LoC_scaled['gamma_tilde_sym6']
            self.rhs_A_sym6 += K_LoC_scaled['A_sym6']
            self.rhs_Gamma_tilde += K_LoC_scaled['Gamma_tilde']
            if slow_update:
                self.rhs_phi += K_LoC_scaled['phi']
                self.rhs_Z += K_LoC_scaled['Z']
                self.rhs_Z_i += K_LoC_scaled['Z_i']

        logger.info("compute_rhs timing", extra={
            "extra_data": {
                "compute_rhs_ms": rhs_timer.elapsed_ms()
            }
        })
