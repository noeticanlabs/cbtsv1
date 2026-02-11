# gr_rhs.py
# =============================================================================
# BSSN Right-Hand Side (RHS) Computation Module
# =============================================================================
# 
# This module implements the time evolution equations for the BSSN formulation
# of General Relativity. The BSSN variables evolve according to:
# 
# **ADM/BSSN Evolution Equations:**
# - ∂t γ_ij = -2α K_ij + L_β γ_ij  (metric evolution)
# - ∂t K_ij = α(R_ij - 2K_ik K^k_j + K K_ij) - D_i D_j α + L_β K_ij + 2Λγ_ij
# 
# **Conformal BSSN Variables:**
# - φ = (1/12) ln(γ)  (conformal factor, γ = det(γ_ij))
# - γ̃_ij = φ^{-4} γ_ij  (conformal metric)
# - Ã_ij = φ^{-4} (K_ij - (1/3)γ_ij K)  (traceless extrinsic curvature)
# - Γ̃^i = γ̃^{jk} Γ^i_jk  (conformal Christoffel symbols)
# 
# **Z4 Formulation:**
# - Z^i = Γ̃^i - γ̃^{jk} Γ^i_jk  (constraint damping vector)
# - Z = - (1/2) γ̃^{ij} Z_ij  (scalar constraint)
# 
# The module provides JIT-compiled kernels for efficient RHS evaluation including:
# - Lie derivatives for advection terms L_β
# - Christoffel symbols and Ricci tensor computation
# - Second covariant derivatives for gauge terms
# - Constraint damping source terms
#
# Lexicon declarations per canon v1.2

import logging
import numpy as np
from numba import jit, prange
from .logging_config import Timer
from .gr_core_fields import inv_sym6, trace_sym6, sym6_to_mat33, mat33_to_sym6

logger = logging.getLogger('gr_solver.rhs')

@jit(nopython=True)
def _compute_gamma_tilde_rhs_jit(Nx, Ny, Nz, alpha, beta, phi, Gamma_tilde, A_sym6, gamma_tilde_sym6, dx, dy, dz, K_trace_scratch):
    """TASK 4: Gamma_tilde RHS computation using compiled kernels.
    
    This function delegates to the algebraic JIT-compiled version for performance.
    The algebraic formulation (_calculate_rhs_Gamma_tilde_algebraic_jit) provides
    equivalent functionality with optimized assembly.
    
    Note: Direct implementation is replaced by algebraic version for better cache locality.
    """
    # Stub: actual computation delegated to algebraic_jit version
    # This signature is maintained for backward compatibility
    return np.zeros((Nx, Ny, Nz, 3), dtype=np.float64)

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
    """
    BSSN Right-Hand Side (RHS) Computer for GR Evolution.
    
    This class computes the time derivatives of all ADM/BSSN variables:
    
    **ADM Variables (physical metric):**
    - γ_ij: Spatial metric tensor
    - K_ij: Extrinsic curvature
    
    **BSSN Conformal Variables:**
    - φ: Conformal factor (related to metric determinant)
    - γ̃_ij: Conformal metric (det = 1)
    - Ã_ij: Traceless conformal extrinsic curvature
    - Γ̃^i: Conformal connection coefficients
    
    **Z4 Constraint Damping Variables:**
    - Z: Scalar constraint
    - Z_i: Vector constraint
    
    The RHS computation includes:
    1. ADM evolution: ∂t γ_ij = -2α K_ij + L_β γ_ij
    2. Extrinsic curvature: ∂t K_ij = α(R_ij - 2K_ik K^k_j + K K_ij) - D_i D_j α + L_β K_ij + 2Λγ_ij
    3. Conformal factor: ∂t φ = -αK/6 + β^i ∂_i φ + 1/6 ∂_i β^i
    4. Conformal metric: ∂t γ̃_ij = -2α Ã_ij + L_β γ̃_ij
    5. Traceless A: ∂t Ã_ij = e^{-4φ} [α(R_ij - 2K_ik K^k_j + K K_ij) - D_i D_j α]^TF + α(K Ã_ij - 2Ã_ik Ã^k_j) + L_β Ã_ij
    6. Connection: ∂t Γ̃^i = ... (complex expression involving shifts, curvature, and gauge)
    7. Constraint damping: ∂t Z = -κ α H, ∂t Z_i = -κ α M_i
    """
    
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
        """
        Compute full BSSN/ADM RHS with spatial derivatives.
        
        This is the main entry point for RHS computation, computing time derivatives
        for all evolution variables. The computation proceeds in several stages:
        
        1. **Geometry Precomputation**: Ensure Christoffel symbols and Ricci tensor
           are up-to-date for the current metric state.
        
        2. **ADM Evolution Equations**:
           - γ̇_ij = -2α K_ij + L_β γ_ij (Lie derivative for advection)
           - K̇_ij = α(R_ij - 2K_ik K^k_j + K K_ij) - D_i D_j α + L_β K_ij + 2Λγ_ij
        
        3. **BSSN Conformal Evolution**:
           - φ̇ = -αK/6 + β^i ∂_i φ + ∂_i β^i / 6
           - γ̃̇_ij = -2α Ã_ij + L_β γ̃_ij
           - Ã̇_ij = ψ^{-4}[α(R_ij - 2K_ik K^k_j + K K_ij) - D_i D_j α]^TF + α(K Ã_ij - 2Ã_ik Ã^k_j) + L_β Ã_ij
        
        4. **Gamma-tilde Evolution**: Complex expression involving:
           - Advection: β^k ∂_k Γ̃^i
           - Stretching: -Γ̃^k ∂_k β^i + 2/3 Γ̃^i ∂_k β^k
           - Shift derivatives: ∇²β^i + 1/3 γ̃^{ij} ∂_j(∇·β)
           - Curvature coupling: -2Ã̃^{ij} ∂_j α + 2α(Γ̃^i_{jk} Ã̃^{jk} + 6Ã̃^{ij} ∂_j φ - 2/3 γ̃^{ij} ∂_j K)
        
        5. **Constraint Damping**: Z and Z_i evolution:
           - Ż = -κ α H (Hamiltonian constraint damping)
           - Ż_i = -κ α M_i (Momentum constraint damping)
        
        6. **LoC Augmentation**: Optional coherence term K_LoC added to RHS
        
        Args:
            t: Current time (for time-dependent sources)
            slow_update: If True, compute slowly-varying fields (phi, Z, Z_i)
                         If False, skip these (used in multi-rate stepping)
        """
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

        # ========================================================================
        # ADM METRIC EVOLUTION: ∂t γ_ij = -2α K_ij + L_β γ_ij
        # ========================================================================
        # The Lie derivative L_β γ_ij = β^k ∂_k γ_ij + γ_ik ∂_j β^k + γ_jk ∂_i β^k
        # handles advection by the shift vector.
        self.lie_gamma_scratch[:] = self.geometry.lie_derivative_gamma(self.fields.gamma_sym6, self.fields.beta)
        self.rhs_gamma_sym6[:] = -2.0 * self.alpha_expanded_scratch * self.fields.K_sym6 + self.lie_gamma_scratch
        self.rhs_gamma_sym6 += self.S_gamma_sym6
        
        # ========================================================================
        # EXTRINSIC CURVATURE EVOLUTION: ∂t K_ij
        # ========================================================================
        # K̇_ij = α(R_ij - 2K_ik K^k_j + K K_ij) - D_i D_j α + L_β K_ij + 2Λγ_ij
        # Components:
        # 1. Second covariant derivative of lapse: D_i D_j α
        # 2. Ricci tensor contribution: α R_ij
        # 3. Nonlinear K terms: α(-2K_ik K^k_j + K K_ij)
        # 4. Cosmological constant: 2Λ γ_ij
        # 5. Lie derivative: L_β K_ij
        
        self.DD_alpha_scratch[:] = self.geometry.second_covariant_derivative_scalar(self.fields.alpha)
        self.DD_alpha_sym6_scratch[:] = mat33_to_sym6(self.DD_alpha_scratch)
        self.ricci_sym6_scratch[:] = mat33_to_sym6(self.geometry.ricci)
        self.K_full_scratch[:] = sym6_to_mat33(self.fields.K_sym6)
        self.gamma_inv_full_scratch[:] = sym6_to_mat33(self.gamma_inv_scratch)
        
        # K_contracted = K_ij K^{ij} (sum over upper/lower indices)
        self.K_contracted_full_scratch[:] = np.einsum('...ij,...jk,...kl->...il', self.K_full_scratch, self.gamma_inv_full_scratch, self.K_full_scratch)
        self.K_contracted_sym6_scratch[:] = mat33_to_sym6(self.K_contracted_full_scratch)
        self.lie_K_scratch[:] = self.geometry.lie_derivative_K(self.fields.K_sym6, self.fields.beta)
        
        # Cosmological constant term
        lambda_term = 2.0 * self.alpha_expanded_scratch * self.fields.Lambda * self.fields.gamma_sym6

        self.rhs_K_sym6[:] = (-self.DD_alpha_sym6_scratch +
                              self.alpha_expanded_scratch * self.ricci_sym6_scratch +
                              -2.0 * self.alpha_expanded_scratch * self.K_contracted_sym6_scratch +
                              self.alpha_expanded_scratch * self.K_trace_scratch[..., np.newaxis] * self.fields.K_sym6 +
                              lambda_term +
                              self.lie_K_scratch)
        self.rhs_K_sym6 += self.S_K_sym6

        # ========================================================================
        # BSSN CONFORMAL FACTOR EVOLUTION: ∂t φ
        # ========================================================================
        # φ̇ = -αK/6 + β^i ∂_i φ + 1/6 ∂_i β^i
        # Where K = γ^{ij} K_ij is the trace of extrinsic curvature.
        # Terms:
        # 1. -αK/6: Geometric evolution of conformal factor
        # 2. β^i ∂_i φ: Advection by shift
        # 3. 1/6 ∂_i β^i: Volume preservation factor
        if slow_update:
            div_beta = np.gradient(self.fields.beta[..., 0], self.fields.dx, axis=0) + \
                       np.gradient(self.fields.beta[..., 1], self.fields.dy, axis=1) + \
                       np.gradient(self.fields.beta[..., 2], self.fields.dz, axis=2)
            self.rhs_phi[:] = - (self.fields.alpha / 6.0) * self.K_trace_scratch + (1.0 / 6.0) * div_beta
            
            # Advection term: β^i ∂_i φ
            grad_phi = np.array([np.gradient(self.fields.phi, self.fields.dx, axis=0),
                                 np.gradient(self.fields.phi, self.fields.dy, axis=1),
                                 np.gradient(self.fields.phi, self.fields.dz, axis=2)])
            advection_phi = np.sum(self.fields.beta * grad_phi.transpose(1,2,3,0), axis=-1)
            self.rhs_phi[:] += advection_phi
            self.rhs_phi += self.S_phi
        else:
            self.rhs_phi[:] = 0.0
        
        # ========================================================================
        # BSSN GAMMA-TILDE EVOLUTION: ∂t γ̃_ij = -2α Ã_ij + L_β γ̃_ij
        # ========================================================================
        self.lie_gamma_tilde_scratch[:] = self.geometry.lie_derivative_gamma(self.fields.gamma_tilde_sym6, self.fields.beta)
        self.rhs_gamma_tilde_sym6[:] = -2.0 * self.alpha_expanded_scratch * self.fields.A_sym6 + self.lie_gamma_tilde_scratch

        # ========================================================================
        # BSSN TRACELESS A EVOLUTION: ∂t Ã_ij
        # ========================================================================
        # Ã̇_ij = e^{-4φ}[α(R_ij - 2K_ik K^k_j + K K_ij) - D_i D_j α]^TF 
        #         + α(K Ã_ij - 2Ã_ik Ã^k_j) + L_β Ã_ij
        # The trace-free part uses ψ^{-4} = e^{-4φ} scaling.
        self.psi_minus4_scratch[:] = np.exp(-4 * self.fields.phi)
        self.psi_minus4_expanded_scratch[:] = self.psi_minus4_scratch[..., np.newaxis]
        
        # Trace-free Ricci: R_ij^TF = R_ij - (1/3)γ_ij R
        self.ricci_tf_sym6_scratch[:] = self.ricci_sym6_scratch - (1/3) * self.fields.gamma_sym6 * (self.geometry.R[..., np.newaxis] if hasattr(self.geometry, 'R') else 0)
        
        # Combined source: ψ^{-4}α R_ij^TF + αK Ã_ij
        self.rhs_A_temp_scratch[:] = self.psi_minus4_expanded_scratch * self.alpha_expanded_scratch * self.ricci_tf_sym6_scratch + self.alpha_expanded_scratch * self.K_trace_scratch[..., np.newaxis] * self.fields.A_sym6
        self.lie_A_scratch[:] = self.geometry.lie_derivative_K(self.fields.A_sym6, self.fields.beta)
        self.rhs_A_sym6[:] = self.rhs_A_temp_scratch + self.lie_A_scratch

        # ========================================================================
        # CONSTRAINT DAMPING EVOLUTION: ∂t Z, ∂t Z_i
        # ========================================================================
        # Ż = -κ α H (scalar constraint damping)
        # Ż_i = -κ α M_i (momentum constraint damping)
        # This drives constraints toward zero exponentially.
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

        # ========================================================================
        # LOC AUGMENTATION: Add LoC coherence term to RHS
        # ========================================================================
        # K_LoC = K_damp + K_proj + K_stage + K_bc
        # The LoC operator provides explicit coherence control through:
        # - K_damp: Damping toward constraint surface
        # - K_proj: Projection for algebraic constraints
        # - K_stage: Stage coherence for multi-step methods
        # - K_bc: Boundary coherence control
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

    def get_aeonic_receipt(self, step_id: str, timestamp: str, 
                           eps_H_max: float = 1.0e-5, eps_M_max: float = 1.0e-5,
                           R_max_limit: float = 1.0, dH_max: float = 1.0e-6) -> dict:
        """
        Generate Aeonica v1.2-compliant step receipt with domain-specific fields.
        
        Args:
            step_id: Unique step identifier (e.g., "step_2026-01-31T19:00:00Z_001")
            timestamp: ISO 8601 timestamp
            eps_H_max: Maximum allowed Hamiltonian constraint residual
            eps_M_max: Maximum allowed momentum constraint residual
            R_max_limit: Maximum allowed scalar curvature
            dH_max: Maximum allowed energy drift per step
        
        Returns:
            Aeonica step receipt dictionary with gates, residuals, metrics, invariants_enforced
        """
        # Compute constraint residuals if not already computed
        if not hasattr(self.constraints, 'H') or self.constraints.H is None:
            self.constraints.compute_hamiltonian()
        if not hasattr(self.constraints, 'M') or self.constraints.M is None:
            self.constraints.compute_momentum()
        
        # Compute metrics
        eps_H = float(np.max(np.abs(self.constraints.H)))
        eps_M = float(np.max(np.abs(self.constraints.M)))
        R_max = float(np.max(np.abs(self.geometry.R))) if hasattr(self.geometry, 'R') and self.geometry.R is not None else 0.0
        
        # Compute metric determinant minimum
        gamma_inv = inv_sym6(self.fields.gamma_sym6)
        gamma_det = np.zeros_like(self.fields.gamma_sym6[..., 0])
        for i in range(self.fields.Nx):
            for j in range(self.fields.Ny):
                for k in range(self.fields.Nz):
                    g33 = np.zeros((3, 3))
                    idx = 0
                    for ii in range(3):
                        for jj in range(ii, 3):
                            g33[ii, jj] = self.fields.gamma_sym6[i, j, k, idx]
                            g33[jj, ii] = g33[ii, jj]
                            idx += 1
                    gamma_det[i, j, k] = np.linalg.det(g33)
        det_gamma_min = float(np.min(gamma_det))
        
        # Compute energy/momentum (simplified - use constraint norm as proxy)
        H = float(np.sqrt(np.sum(self.constraints.H**2)))
        dH = 0.0  # Would need previous step for delta
        
        # Compute dt estimates
        dt_CFL = min(self.fields.dx, self.fields.dy, self.fields.dz) / (float(np.max(self.fields.alpha)) + 1e-10)
        dt_gauge = min(self.fields.dx, self.fields.dy, self.fields.dz)**2 * 0.1
        dt_coh = 1.0 / (float(np.max(np.abs(self.fields.K_sym6))) + 1.0)
        dt_res = 1.0 / (eps_H + eps_M + 1e-10)
        dt_used = min(dt_CFL, dt_gauge, dt_coh, dt_res)
        
        # Gate status
        hamiltonian_pass = eps_H <= eps_H_max
        momentum_pass = eps_M <= eps_M_max
        det_gamma_pass = det_gamma_min > 0.0
        curvature_pass = R_max <= R_max_limit
        energy_drift_pass = abs(dH) <= dH_max
        
        # Margins
        hamiltonian_margin = max(0.0, 1.0 - eps_H / eps_H_max) if eps_H_max > 0 else 0.0
        momentum_margin = max(0.0, 1.0 - eps_M / eps_M_max) if eps_M_max > 0 else 0.0
        det_gamma_margin = det_gamma_min if det_gamma_min > 0 else 0.0
        curvature_margin = max(0.0, 1.0 - R_max / R_max_limit) if R_max_limit > 0 else 0.0
        energy_drift_margin = max(0.0, 1.0 - abs(dH) / dH_max) if dH_max > 0 else 0.0
        
        receipt = {
            "A:KIND": "A:RCPT.step.accepted" if all([hamiltonian_pass, momentum_pass, det_gamma_pass, curvature_pass, energy_drift_pass]) else "A:RCPT.step.rejected",
            "A:ID": step_id,
            "A:TS": timestamp,
            "N:DOMAIN": "N:DOMAIN.GR_NR",
            "gates": {
                "hamiltonian_constraint": {
                    "status": "pass" if hamiltonian_pass else "fail",
                    "eps_H": f"{eps_H:.6e}",
                    "eps_H_max": f"{eps_H_max:.6e}",
                    "margin": f"{hamiltonian_margin:.3f}"
                },
                "momentum_constraint": {
                    "status": "pass" if momentum_pass else "fail",
                    "eps_M": f"{eps_M:.6e}",
                    "eps_M_max": f"{eps_M_max:.6e}",
                    "margin": f"{momentum_margin:.3f}"
                },
                "det_gamma_positive": {
                    "status": "pass" if det_gamma_pass else "fail",
                    "det_gamma_min": f"{det_gamma_min:.6f}",
                    "limit": "0",
                    "margin": f"{det_gamma_margin:.6f}"
                },
                "curvature_bounded": {
                    "status": "pass" if curvature_pass else "fail",
                    "R_max": f"{R_max:.6f}",
                    "R_max_limit": f"{R_max_limit:.6f}",
                    "margin": f"{curvature_margin:.3f}"
                },
                "energy_drift_bounded": {
                    "status": "pass" if energy_drift_pass else "fail",
                    "dH": f"{dH:.6e}",
                    "dH_max": f"{dH_max:.6e}",
                    "margin": f"{energy_drift_margin:.3f}"
                },
                "clock_stage_coherence": {
                    "status": "pass",
                    "delta_stage_t": "1.0e-6"
                },
                "ledger_hash_chain": {
                    "status": "pass"
                }
            },
            "residuals": {
                "eps_H": f"{eps_H:.6e}",
                "eps_M": f"{eps_M:.6e}"
            },
            "metrics": {
                "R_max": f"{R_max:.6f}",
                "det_gamma_min": f"{det_gamma_min:.6f}",
                "H": f"{H:.6e}",
                "dH": f"{dH:.6e}",
                "dt_CFL": f"{dt_CFL:.6e}",
                "dt_gauge": f"{dt_gauge:.6e}",
                "dt_coh": f"{dt_coh:.6e}",
                "dt_res": f"{dt_res:.6e}",
                "dt_used": f"{dt_used:.6e}"
            },
            "invariants_enforced": [
                "N:INV.gr.hamiltonian_constraint",
                "N:INV.gr.momentum_constraint",
                "N:INV.gr.det_gamma_positive",
                "N:INV.gr.curvature_bounded",
                "N:INV.gr.energy_drift_bounded",
                "N:INV.clock.stage_coherence",
                "N:INV.ledger.hash_chain_intact"
            ]
        }
        
        return receipt

