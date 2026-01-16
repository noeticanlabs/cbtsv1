# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time"
]
LEXICON_SYMBOLS = {
    "\\Psi": "UFE_state",
    "B": "UFE_baseline",
    "K": "UFE_coherence"
}

import numpy as np
import hashlib
import logging
from .logging_config import Timer, array_stats
from .gr_core_fields import inv_sym6, trace_sym6, sym6_to_mat33, mat33_to_sym6
from stepper_contract_memory import StepperContractWithMemory

logger = logging.getLogger('gr_solver.stepper')

class GRStepper:
    def __init__(self, fields, geometry, constraints, gauge, memory_contract=None, phaseloom=None, aeonic_mode=True):
        self.fields = fields
        self.geometry = geometry
        self.constraints = constraints
        self.gauge = gauge
        self.dt_applied = 0.0
        self.damping_enabled = True
        self.lambda_val = 0.0
        self.dealiasing_enabled = True
        self.sources_func = None
        self.memory = StepperContractWithMemory(memory_contract, phaseloom, max_attempts=20, dt_floor=1e-10) if memory_contract and phaseloom else None
        self.aeonic_mode = aeonic_mode

        # Multi-rate stepping parameters
        self.slow_fields = ['phi', 'Z', 'Z_i']
        self.slow_rate = 5  # Update slow fields every 5 steps
        self.step_count = 0

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
            self.lie_gamma_tilde_scratch = np.zeros_like(self.fields.gamma_sym6)
            self.psi_minus4_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz))
            self.psi_minus4_expanded_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 6))
            self.ricci_tf_sym6_scratch = np.zeros_like(self.fields.gamma_sym6)
            self.rhs_A_temp_scratch = np.zeros_like(self.fields.gamma_sym6)
            self.lie_A_scratch = np.zeros_like(self.fields.gamma_sym6)

    def step_ufe(self, dt, t=0.0):
        """RK4 step for UFE."""
        with Timer("step_ufe") as timer:
            self.dt_applied = dt  # Set dt_applied as source of truth

            # Log initial constraint residuals before stepping
            self.constraints.compute_residuals()
            logger.debug("Starting UFE step", extra={
                "extra_data": {
                    "dt": dt,
                    "t": t,
                    "eps_H_initial": float(self.constraints.eps_H),
                    "eps_M_initial": float(self.constraints.eps_M)
                }
            })

            slow_update = (self.step_count % self.slow_rate == 0)

            # Save initial state
            u0_gamma = self.fields.gamma_sym6.copy()
            u0_K = self.fields.K_sym6.copy()
            u0_phi = self.fields.phi.copy()
            u0_gamma_tilde = self.fields.gamma_tilde_sym6.copy()
            u0_A = self.fields.A_sym6.copy()
            u0_Z = self.fields.Z.copy()
            u0_Z_i = self.fields.Z_i.copy()

            # Stage 1
            self.compute_rhs(t, slow_update)
            k1_gamma = self.rhs_gamma_sym6.copy()
            k1_K = self.rhs_K_sym6.copy()
            k1_phi = self.rhs_phi.copy()
            k1_gamma_tilde = self.rhs_gamma_tilde_sym6.copy()
            k1_A = self.rhs_A_sym6.copy()
            k1_Z = self.rhs_Z.copy()
            k1_Z_i = self.rhs_Z_i.copy()

            # Stage 2: u + dt/2 * k1
            self.fields.gamma_sym6 = u0_gamma + (dt/2) * k1_gamma
            self.fields.K_sym6 = u0_K + (dt/2) * k1_K
            self.fields.phi = u0_phi + (dt/2) * k1_phi
            self.fields.gamma_tilde_sym6 = u0_gamma_tilde + (dt/2) * k1_gamma_tilde
            self.fields.A_sym6 = u0_A + (dt/2) * k1_A
            self.fields.Z = u0_Z + (dt/2) * k1_Z
            self.fields.Z_i = u0_Z_i + (dt/2) * k1_Z_i
            self.geometry.compute_christoffels()
            self.geometry.compute_ricci()
            self.geometry.compute_scalar_curvature()
            self.compute_rhs(t + dt/2, slow_update)
            k2_gamma = self.rhs_gamma_sym6.copy()
            k2_K = self.rhs_K_sym6.copy()
            k2_phi = self.rhs_phi.copy()
            k2_gamma_tilde = self.rhs_gamma_tilde_sym6.copy()
            k2_A = self.rhs_A_sym6.copy()
            k2_Z = self.rhs_Z.copy()
            k2_Z_i = self.rhs_Z_i.copy()

            # Stage 3: u + dt/2 * k2
            self.fields.gamma_sym6 = u0_gamma + (dt/2) * k2_gamma
            self.fields.K_sym6 = u0_K + (dt/2) * k2_K
            self.fields.phi = u0_phi + (dt/2) * k2_phi
            self.fields.gamma_tilde_sym6 = u0_gamma_tilde + (dt/2) * k2_gamma_tilde
            self.fields.A_sym6 = u0_A + (dt/2) * k2_A
            self.fields.Z = u0_Z + (dt/2) * k2_Z
            self.fields.Z_i = u0_Z_i + (dt/2) * k2_Z_i
            self.geometry.compute_christoffels()
            self.geometry.compute_ricci()
            self.geometry.compute_scalar_curvature()
            self.compute_rhs(t + dt/2, slow_update)
            k3_gamma = self.rhs_gamma_sym6.copy()
            k3_K = self.rhs_K_sym6.copy()
            k3_phi = self.rhs_phi.copy()
            k3_gamma_tilde = self.rhs_gamma_tilde_sym6.copy()
            k3_A = self.rhs_A_sym6.copy()
            k3_Z = self.rhs_Z.copy()
            k3_Z_i = self.rhs_Z_i.copy()

            # Stage 4: u + dt * k3
            self.fields.gamma_sym6 = u0_gamma + dt * k3_gamma
            self.fields.K_sym6 = u0_K + dt * k3_K
            self.fields.phi = u0_phi + dt * k3_phi
            self.fields.gamma_tilde_sym6 = u0_gamma_tilde + dt * k3_gamma_tilde
            self.fields.A_sym6 = u0_A + dt * k3_A
            self.fields.Z = u0_Z + dt * k3_Z
            self.fields.Z_i = u0_Z_i + dt * k3_Z_i
            self.geometry.compute_christoffels()
            self.geometry.compute_ricci()
            self.geometry.compute_scalar_curvature()
            self.compute_rhs(t + dt, slow_update)
            k4_gamma = self.rhs_gamma_sym6.copy()
            k4_K = self.rhs_K_sym6.copy()
            k4_phi = self.rhs_phi.copy()
            k4_gamma_tilde = self.rhs_gamma_tilde_sym6.copy()
            k4_A = self.rhs_A_sym6.copy()
            k4_Z = self.rhs_Z.copy()
            k4_Z_i = self.rhs_Z_i.copy()

            # Final update
            self.fields.gamma_sym6 = u0_gamma + (dt/6) * (k1_gamma + 2*k2_gamma + 2*k3_gamma + k4_gamma)
            self.fields.K_sym6 = u0_K + (dt/6) * (k1_K + 2*k2_K + 2*k3_K + k4_K)
            self.fields.phi = u0_phi + (dt/6) * (k1_phi + 2*k2_phi + 2*k3_phi + k4_phi)
            self.fields.gamma_tilde_sym6 = u0_gamma_tilde + (dt/6) * (k1_gamma_tilde + 2*k2_gamma_tilde + 2*k3_gamma_tilde + k4_gamma_tilde)
            self.fields.A_sym6 = u0_A + (dt/6) * (k1_A + 2*k2_A + 2*k3_A + k4_A)
            self.fields.Z = u0_Z + (dt/6) * (k1_Z + 2*k2_Z + 2*k3_Z + k4_Z)
            self.fields.Z_i = u0_Z_i + (dt/6) * (k1_Z_i + 2*k2_Z_i + 2*k3_Z_i + k4_Z_i)

            # Recompute geometry for final state
            self.geometry.compute_christoffels()
            self.geometry.compute_ricci()
            self.geometry.compute_scalar_curvature()

            # Apply constraint damping (projection effects)
            self.apply_damping()
            self.constraints.compute_residuals()
            # Evolve gauge
            self.gauge.evolve_lapse(dt)
            self.gauge.evolve_shift(dt)

            # Increment step count for multi-rate
            self.step_count += 1

            logger.debug("UFE step completed", extra={
                "extra_data": {
                    "execution_time_ms": timer.elapsed_ms(),
                    "eps_H_final": float(self.constraints.eps_H),
                    "eps_M_final": float(self.constraints.eps_M),
                    "field_stats": {
                        "gamma_sym6": array_stats(self.fields.gamma_sym6, "gamma_sym6"),
                        "K_sym6": array_stats(self.fields.K_sym6, "K_sym6"),
                        "alpha": array_stats(self.fields.alpha, "alpha")
                    }
                }
            })

    def compute_rhs(self, t=0.0, slow_update=True):
        """Compute full ADM RHS B with spatial derivatives."""
        if self.aeonic_mode:
            rhs_gamma_sym6 = self.rhs_gamma_sym6
            rhs_K_sym6 = self.rhs_K_sym6
            rhs_phi = self.rhs_phi
            rhs_gamma_tilde_sym6 = self.rhs_gamma_tilde_sym6
            rhs_A_sym6 = self.rhs_A_sym6
            rhs_Gamma_tilde = self.rhs_Gamma_tilde
            rhs_Z = self.rhs_Z
            rhs_Z_i = self.rhs_Z_i
            S_gamma_tilde_sym6 = self.S_gamma_tilde_sym6
            S_A_sym6 = self.S_A_sym6
            S_phi = self.S_phi
            S_Gamma_tilde = self.S_Gamma_tilde
            S_Z = self.S_Z
            S_Z_i = self.S_Z_i
            gamma_inv_scratch = self.gamma_inv_scratch
            K_trace_scratch = self.K_trace_scratch
            alpha_expanded_scratch = self.alpha_expanded_scratch
            alpha_expanded_33_scratch = self.alpha_expanded_33_scratch
            lie_gamma_scratch = self.lie_gamma_scratch
            DD_alpha_scratch = self.DD_alpha_scratch
            DD_alpha_sym6_scratch = self.DD_alpha_sym6_scratch
            ricci_sym6_scratch = self.ricci_sym6_scratch
            K_full_scratch = self.K_full_scratch
            gamma_inv_full_scratch = self.gamma_inv_full_scratch
            K_contracted_full_scratch = self.K_contracted_full_scratch
            K_contracted_sym6_scratch = self.K_contracted_sym6_scratch
            lie_K_scratch = self.lie_K_scratch
            lie_gamma_tilde_scratch = self.lie_gamma_tilde_scratch
            psi_minus4_scratch = self.psi_minus4_scratch
            psi_minus4_expanded_scratch = self.psi_minus4_expanded_scratch
            ricci_tf_sym6_scratch = self.ricci_tf_sym6_scratch
            rhs_A_temp_scratch = self.rhs_A_temp_scratch
            lie_A_scratch = self.lie_A_scratch
        else:
            rhs_gamma_sym6 = np.zeros_like(self.fields.gamma_sym6)
            rhs_K_sym6 = np.zeros_like(self.fields.K_sym6)
            rhs_phi = np.zeros_like(self.fields.phi)
            rhs_gamma_tilde_sym6 = np.zeros_like(self.fields.gamma_tilde_sym6)
            rhs_A_sym6 = np.zeros_like(self.fields.A_sym6)
            rhs_Gamma_tilde = np.zeros_like(self.fields.Gamma_tilde)
            rhs_Z = np.zeros_like(self.fields.Z)
            rhs_Z_i = np.zeros_like(self.fields.Z_i)
            S_gamma_tilde_sym6 = np.zeros_like(self.fields.gamma_tilde_sym6)
            S_A_sym6 = np.zeros_like(self.fields.A_sym6)
            S_phi = np.zeros_like(self.fields.phi)
            S_Gamma_tilde = np.zeros_like(self.fields.Gamma_tilde)
            S_Z = np.zeros_like(self.fields.Z)
            S_Z_i = np.zeros_like(self.fields.Z_i)
            gamma_inv_scratch = np.zeros_like(self.fields.gamma_sym6)
            K_trace_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz))
            alpha_expanded_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 6))
            alpha_expanded_33_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
            lie_gamma_scratch = np.zeros_like(self.fields.gamma_sym6)
            DD_alpha_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
            DD_alpha_sym6_scratch = np.zeros_like(self.fields.gamma_sym6)
            ricci_sym6_scratch = np.zeros_like(self.fields.gamma_sym6)
            K_full_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
            gamma_inv_full_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
            K_contracted_full_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
            K_contracted_sym6_scratch = np.zeros_like(self.fields.gamma_sym6)
            lie_K_scratch = np.zeros_like(self.fields.gamma_sym6)
            lie_gamma_tilde_scratch = np.zeros_like(self.fields.gamma_sym6)
            psi_minus4_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz))
            psi_minus4_expanded_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 6))
            ricci_tf_sym6_scratch = np.zeros_like(self.fields.gamma_sym6)
            rhs_A_temp_scratch = np.zeros_like(self.fields.gamma_sym6)
            lie_A_scratch = np.zeros_like(self.fields.gamma_sym6)

        # Set sources if func provided
        if self.sources_func is not None:
            sources = self.sources_func(t)
            S_gamma_tilde_sym6[:] = sources['S_gamma_tilde_sym6']
            S_A_sym6[:] = sources['S_A_sym6']
            S_phi[:] = sources['S_phi']
            S_Gamma_tilde[:] = sources['S_Gamma_tilde']
            S_Z[:] = sources['S_Z']
            S_Z_i[:] = sources['S_Z_i']

        # Ensure geometry is computed
        if not hasattr(self.geometry, 'ricci') or self.geometry.ricci is None:
            self.geometry.compute_ricci()

        # Compute K = gamma^{ij} K_ij
        gamma_inv_scratch[:] = inv_sym6(self.fields.gamma_sym6)
        K_trace_scratch[:] = trace_sym6(self.fields.K_sym6, gamma_inv_scratch)

        alpha_expanded_scratch[:] = self.fields.alpha[..., np.newaxis]  # (Nx,Ny,Nz,1) -> (Nx,Ny,Nz,6)
        alpha_expanded_33_scratch[:] = self.fields.alpha[..., np.newaxis, np.newaxis]  # (Nx,Ny,Nz,1,1) -> (Nx,Ny,Nz,3,3)

        # ADM ∂t gamma_ij = -2 α K_ij + L_β γ_ij
        lie_gamma_scratch[:] = self.geometry.lie_derivative_gamma(self.fields.gamma_sym6, self.fields.beta)
        rhs_gamma_sym6[:] = -2.0 * alpha_expanded_scratch * self.fields.K_sym6 + lie_gamma_scratch

        # ADM ∂t K_ij = -D_i D_j α + α R_ij - 2 α K_ik γ^{kl} K_lj + α K K_ij + L_β K_ij
        # D_i D_j α
        DD_alpha_scratch[:] = self.geometry.second_covariant_derivative_scalar(self.fields.alpha)
        DD_alpha_sym6_scratch[:] = mat33_to_sym6(DD_alpha_scratch)

        # R_ij in sym6 form
        ricci_sym6_scratch[:] = mat33_to_sym6(self.geometry.ricci)

        # K_ik γ^{kl} K_lj = (K γ^{-1} K)_ij
        K_full_scratch[:] = sym6_to_mat33(self.fields.K_sym6)
        gamma_inv_full_scratch[:] = sym6_to_mat33(gamma_inv_scratch)

        # K^{kl} = γ^{ki} γ^{lj} K_ij, but for contraction K_ik γ^{kl} K_lj
        K_contracted_full_scratch[:] = np.einsum('...ij,...jk,...kl->...il', K_full_scratch, gamma_inv_full_scratch, K_full_scratch)

        K_contracted_sym6_scratch[:] = mat33_to_sym6(K_contracted_full_scratch)

        lie_K_scratch[:] = self.geometry.lie_derivative_K(self.fields.K_sym6, self.fields.beta)

        rhs_K_sym6[:] = (-DD_alpha_sym6_scratch +
                              alpha_expanded_scratch * ricci_sym6_scratch +
                              -2.0 * alpha_expanded_scratch * K_contracted_sym6_scratch +
                              alpha_expanded_scratch * K_trace_scratch[..., np.newaxis] * self.fields.K_sym6 +
                              lie_K_scratch)

        # BSSN ∂_0 φ = - (α/6) K + (1/6) ∂_k β^k
        # ∂_t φ = ∂_0 φ + β^k ∂_k φ
        div_beta = np.gradient(self.fields.beta[..., 0], self.fields.dx, axis=0) + \
                   np.gradient(self.fields.beta[..., 1], self.fields.dy, axis=1) + \
                   np.gradient(self.fields.beta[..., 2], self.fields.dz, axis=2)
        grad_beta = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
        for k in range(3):
            grad_beta[..., k, 0] = np.gradient(self.fields.beta[..., k], self.fields.dx, axis=0)
            grad_beta[..., k, 1] = np.gradient(self.fields.beta[..., k], self.fields.dy, axis=1)
            grad_beta[..., k, 2] = np.gradient(self.fields.beta[..., k], self.fields.dz, axis=2)
        rhs_phi[:] = - (self.fields.alpha / 6.0) * K_trace_scratch + (1.0 / 6.0) * div_beta
        # Add advection term β^k ∂_k φ
        grad_phi = np.array([np.gradient(self.fields.phi, self.fields.dx, axis=0),
                             np.gradient(self.fields.phi, self.fields.dy, axis=1),
                             np.gradient(self.fields.phi, self.fields.dz, axis=2)])
        advection_phi = np.sum(self.fields.beta * grad_phi.transpose(1,2,3,0), axis=-1)
        rhs_phi[:] += advection_phi
        # Full BSSN Gamma_tilde evolution: ∂_t \tilde Γ^i = A + B + C + D
        # Precompute gamma_tilde_inv
        gamma_tilde_inv = inv_sym6(self.fields.gamma_tilde_sym6)
        gamma_tilde_inv_full = sym6_to_mat33(gamma_tilde_inv)

        # Precompute A_tilde_uu (contravariant)
        A_full = sym6_to_mat33(self.fields.A_sym6)
        A_tilde_uu = np.einsum('...ij,...jk->...ik', gamma_tilde_inv_full, A_full)

        # Precompute Christoffel tildeGamma^i_{jk}
        christoffel_tilde_udd = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3, 3))
        self.geometry.compute_christoffels_for_metric(self.fields.gamma_tilde_sym6, christoffel_tilde_udd)

        # Gradients
        dalpha_x = np.gradient(self.fields.alpha, self.fields.dx, axis=0)
        dalpha_y = np.gradient(self.fields.alpha, self.fields.dy, axis=1)
        dalpha_z = np.gradient(self.fields.alpha, self.fields.dz, axis=2)
        dphi_x = np.gradient(self.fields.phi, self.fields.dx, axis=0)
        dphi_y = np.gradient(self.fields.phi, self.fields.dy, axis=1)
        dphi_z = np.gradient(self.fields.phi, self.fields.dz, axis=2)
        dK_x = np.gradient(K_trace_scratch, self.fields.dx, axis=0)
        dK_y = np.gradient(K_trace_scratch, self.fields.dy, axis=1)
        dK_z = np.gradient(K_trace_scratch, self.fields.dz, axis=2)

        # Shift gradients
        dbeta = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
        for k in range(3):
            dbeta[..., k, 0] = np.gradient(self.fields.beta[..., k], self.fields.dx, axis=0)
            dbeta[..., k, 1] = np.gradient(self.fields.beta[..., k], self.fields.dy, axis=1)
            dbeta[..., k, 2] = np.gradient(self.fields.beta[..., k], self.fields.dz, axis=2)

        # Second derivatives for C
        lap_beta = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3))
        for i in range(3):
            fxx = np.gradient(np.gradient(self.fields.beta[..., i], self.fields.dx, axis=0), self.fields.dx, axis=0)
            fyy = np.gradient(np.gradient(self.fields.beta[..., i], self.fields.dy, axis=1), self.fields.dy, axis=1)
            fzz = np.gradient(np.gradient(self.fields.beta[..., i], self.fields.dz, axis=2), self.fields.dz, axis=2)
            lap_beta[..., i] = fxx + fyy + fzz

        d_div_beta_x = np.gradient(div_beta, self.fields.dx, axis=0)
        d_div_beta_y = np.gradient(div_beta, self.fields.dy, axis=1)
        d_div_beta_z = np.gradient(div_beta, self.fields.dz, axis=2)

        # Gamma gradients
        dGamma = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
        for i in range(3):
            dGamma[..., i, 0] = np.gradient(self.fields.Gamma_tilde[..., i], self.fields.dx, axis=0)
            dGamma[..., i, 1] = np.gradient(self.fields.Gamma_tilde[..., i], self.fields.dy, axis=1)
            dGamma[..., i, 2] = np.gradient(self.fields.Gamma_tilde[..., i], self.fields.dz, axis=2)

        rhs_Gamma_tilde = np.zeros_like(self.fields.Gamma_tilde)

        # A: advection
        rhs_Gamma_tilde += np.einsum('...k,...ik->...i', self.fields.beta, dGamma)

        # B: stretching
        Gamma_dot_grad_beta = np.einsum('...k,...ik->...i', self.fields.Gamma_tilde, dbeta)
        rhs_Gamma_tilde += -Gamma_dot_grad_beta + (2.0/3.0) * self.fields.Gamma_tilde * div_beta[..., np.newaxis]

        # C: shift second-derivatives (approximated)
        d_div_beta = np.array([d_div_beta_x, d_div_beta_y, d_div_beta_z]).transpose(1,2,3,0)
        rhs_Gamma_tilde += lap_beta + (1.0/3.0) * np.einsum('...ij,...j->...i', gamma_tilde_inv_full, d_div_beta)

        # D: lapse/curvature
        # -2 A^{ij} d_j alpha
        dalpha = np.array([dalpha_x, dalpha_y, dalpha_z]).transpose(1,2,3,0)
        rhs_Gamma_tilde += -2.0 * np.einsum('...ij,...j->...i', A_tilde_uu, dalpha)

        # 2 alpha (Gamma^i_{jk} A^{jk} + 6 A^{ij} d_j phi - (2/3) gamma^{ij} d_j K)
        GammaA = np.einsum('...ijk,...jk->...i', christoffel_tilde_udd, A_tilde_uu)

        dphi = np.array([dphi_x, dphi_y, dphi_z]).transpose(1,2,3,0)
        A_dphi = np.einsum('...ij,...j->...i', A_tilde_uu, dphi)

        dK = np.array([dK_x, dK_y, dK_z]).transpose(1,2,3,0)
        gamma_dK = np.einsum('...ij,...j->...i', gamma_tilde_inv_full, dK)

        rhs_Gamma_tilde += 2.0 * self.fields.alpha[..., np.newaxis] * (GammaA + 6.0 * A_dphi - (2.0/3.0) * gamma_dK)

        # BSSN ∂t γ̃_ij = -2 α A_ij + L_β γ̃_ij
        lie_gamma_tilde_scratch[:] = self.geometry.lie_derivative_gamma(self.fields.gamma_tilde_sym6, self.fields.beta)
        rhs_gamma_tilde_sym6[:] = -2.0 * alpha_expanded_scratch * self.fields.A_sym6 + lie_gamma_tilde_scratch

        # BSSN ∂_0 A_ij = e^{-4φ} [-D_i D_j α + α R_ij]^TF + α (K A_ij - 2 A_il A^l_j) + 2 A_k(i ∂_j) β^k - (2/3) A_ij ∂_k β^k
        # Simplified version, ignoring matter for now
        psi_minus4_scratch[:] = np.exp(-4 * self.fields.phi)
        psi_minus4_expanded_scratch[:] = psi_minus4_scratch[..., np.newaxis]
        ricci_tf_sym6_scratch[:] = ricci_sym6_scratch - (1/3) * self.fields.gamma_sym6 * (self.geometry.R[..., np.newaxis] if hasattr(self.geometry, 'R') else 0)  # Approximate

        rhs_A_temp_scratch[:] = psi_minus4_expanded_scratch * alpha_expanded_scratch * ricci_tf_sym6_scratch + alpha_expanded_scratch * K_trace_scratch[..., np.newaxis] * self.fields.A_sym6

        # Add shift terms for A_ij
        # A_full = sym6_to_mat33(self.fields.A_sym6)
        # shift_term_A = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
        # for i in range(3):
        #     for j in range(3):
        #         for k in range(3):
        #             shift_term_A[..., i, j] += A_full[..., k, i] * grad_beta[..., k, j] + A_full[..., k, j] * grad_beta[..., k, i]
        # gauge_term_A = - (2/3.0) * A_full * div_beta[..., np.newaxis, np.newaxis]
        # rhs_A_temp_scratch += mat33_to_sym6(shift_term_A + gauge_term_A)

        lie_A_scratch[:] = self.geometry.lie_derivative_K(self.fields.A_sym6, self.fields.beta)
        rhs_A_sym6[:] = rhs_A_temp_scratch + lie_A_scratch

        # RHS for constraint damping
        # For Z: ∂t Z = - kappa alpha H
        self.constraints.compute_hamiltonian()
        kappa = 1.0
        rhs_Z[:] = -kappa * self.fields.alpha * self.constraints.H
        # For Z_i: ∂t Z_i = - kappa alpha M_i
        self.constraints.compute_momentum()
        rhs_Z_i[:] = -kappa * self.fields.alpha[:, :, :, np.newaxis] * self.constraints.M

        # Add sources
        rhs_gamma_tilde_sym6 += S_gamma_tilde_sym6
        rhs_A_sym6 += S_A_sym6
        rhs_Gamma_tilde += S_Gamma_tilde
        rhs_phi += S_phi
        rhs_Z += S_Z
        rhs_Z_i += S_Z_i

        if not self.aeonic_mode:
            self.rhs_gamma_sym6 = rhs_gamma_sym6
            self.rhs_K_sym6 = rhs_K_sym6
            self.rhs_phi = rhs_phi
            self.rhs_gamma_tilde_sym6 = rhs_gamma_tilde_sym6
            self.rhs_A_sym6 = rhs_A_sym6
            self.rhs_Gamma_tilde = rhs_Gamma_tilde
            self.rhs_Z = rhs_Z
            self.rhs_Z_i = rhs_Z_i

    def apply_damping(self):
        """Apply constraint damping: reduce constraint violations."""
        if not self.damping_enabled:
            logger.debug("Damping disabled")
            return
        # Simple damping: decay the extrinsic curvature to reduce H
        # Note: For high-frequency gauge pulses, this is insufficient (eps_H ~ 2.5e-3). Full Z4 needed.
        decay_factor = np.exp(-self.lambda_val)  # Use lambda_val as damping rate
        old_max_K = np.max(np.abs(self.fields.K_sym6))
        self.fields.K_sym6 *= decay_factor
        new_max_K = np.max(np.abs(self.fields.K_sym6))
        logger.debug("Applied constraint damping", extra={
            "extra_data": {
                "lambda_val": self.lambda_val,
                "decay_factor": decay_factor,
                "K_max_before": old_max_K,
                "K_max_after": new_max_K
            }
        })


