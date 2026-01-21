"""
Test 1 MMS Lite
"""

import numpy as np
import logging
from gr_solver import GRSolver
from gr_solver.gr_core_fields import SYM6_IDX

class Test1MmsLite:
    def __init__(self, gr_solver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def set_mms(self, t, N, dx, dy, dz, L=16.0):
        x = np.arange(N) * dx
        y = np.arange(N) * dy
        z = np.arange(N) * dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        kx = 2 * np.pi / L
        ky = 2 * np.pi / L
        kz = 2 * np.pi / L
        kdotx = kx * X + ky * Y + kz * Z
        omega = 1.0
        eps = 1e-3
        gamma_sym6 = np.zeros((N, N, N, 6))
        gamma_sym6[..., 0] = 1 + eps * np.sin(kdotx) * np.sin(omega * t)  # xx
        gamma_sym6[..., 3] = 1 + eps * np.sin(kdotx) * np.sin(omega * t)  # yy
        gamma_sym6[..., 5] = 1 + eps * np.sin(kdotx) * np.sin(omega * t)  # zz
        K_sym6 = np.zeros((N, N, N, 6))
        K_sym6[..., 0] = eps * np.cos(kdotx) * np.sin(omega * t)
        K_sym6[..., 3] = eps * np.cos(kdotx) * np.sin(omega * t)
        K_sym6[..., 5] = eps * np.cos(kdotx) * np.sin(omega * t)
        alpha = np.ones((N, N, N))
        beta = np.zeros((N, N, N, 3))
        return gamma_sym6, K_sym6, alpha, beta

    def compute_dt_mms(self, t, N, dx, dy, dz, L=16.0):
        x = np.arange(N) * dx
        y = np.arange(N) * dy
        z = np.arange(N) * dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        kx = 2 * np.pi / L
        ky = 2 * np.pi / L
        kz = 2 * np.pi / L
        kdotx = kx * X + ky * Y + kz * Z
        omega = 1.0
        eps = 1e-3
        dt_gamma_sym6 = np.zeros((N, N, N, 6))
        dt_gamma_sym6[..., 0] = eps * omega * np.cos(omega * t) * np.sin(kdotx)
        dt_gamma_sym6[..., 3] = eps * omega * np.cos(omega * t) * np.sin(kdotx)
        dt_gamma_sym6[..., 5] = eps * omega * np.cos(omega * t) * np.sin(kdotx)
        dt_K_sym6 = np.zeros((N, N, N, 6))
        dt_K_sym6[..., 0] = eps * omega * np.cos(omega * t) * np.cos(kdotx)
        dt_K_sym6[..., 3] = eps * omega * np.cos(omega * t) * np.cos(kdotx)
        dt_K_sym6[..., 5] = eps * omega * np.cos(omega * t) * np.cos(kdotx)
        dt_alpha = np.zeros((N, N, N))
        dt_beta = np.zeros((N, N, N, 3))
        return dt_gamma_sym6, dt_K_sym6, dt_alpha, dt_beta

    def compute_Gamma_tilde_mms(self, t, N, dx, dy, dz, L=16.0):
        x = np.arange(N) * dx
        y = np.arange(N) * dy
        z = np.arange(N) * dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        kx = 2 * np.pi / L
        ky = 2 * np.pi / L
        kz = 2 * np.pi / L
        kdotx = kx * X + ky * Y + kz * Z
        omega = 1.0
        eps = 1e-3
        
        sin_k = np.sin(kdotx)
        cos_k = np.cos(kdotx)
        sin_wt = np.sin(omega * t)
        
        F = eps * sin_k * sin_wt
        
        Gamma_tilde = np.zeros((N, N, N, 3))
        ks = [kx, ky, kz]
        for i in range(3):
            ki = ks[i]
            di_F = eps * ki * cos_k * sin_wt
            # Gamma^i = -0.5 * (1+F)^-2 * di_F
            Gamma_tilde[..., i] = -0.5 * (1 + F)**(-2) * di_F
            
        return Gamma_tilde

    def compute_dt_Gamma_tilde_mms(self, t, N, dx, dy, dz, L=16.0):
        x = np.arange(N) * dx
        y = np.arange(N) * dy
        z = np.arange(N) * dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        kx = 2 * np.pi / L
        ky = 2 * np.pi / L
        kz = 2 * np.pi / L
        kdotx = kx * X + ky * Y + kz * Z
        omega = 1.0
        eps = 1e-3
        
        sin_k = np.sin(kdotx)
        cos_k = np.cos(kdotx)
        sin_wt = np.sin(omega * t)
        cos_wt = np.cos(omega * t)
        
        F = eps * sin_k * sin_wt
        dt_F = eps * omega * sin_k * cos_wt
        
        dt_Gamma_tilde = np.zeros((N, N, N, 3))
        ks = [kx, ky, kz]
        for i in range(3):
            ki = ks[i]
            di_F = eps * ki * cos_k * sin_wt
            dt_di_F = eps * ki * omega * cos_k * cos_wt
            
            # dt_Gamma^i = (1+F)^-3 * dt_F * di_F - 0.5 * (1+F)^-2 * dt_di_F
            term1 = (1 + F)**(-3) * dt_F * di_F
            term2 = 0.5 * (1 + F)**(-2) * dt_di_F
            dt_Gamma_tilde[..., i] = term1 - term2
            
        return dt_Gamma_tilde

    def compute_full_gamma_driver_rhs(self, solver):
        """
        Implements the full BSSN evolution equation for Gamma_tilde:
        dt_Gamma^i = -2*A^ij*dj_alpha + 2*alpha*(Gamma^i_jk*A^jk + 6*A^ij*dj_phi - (2/3)*gamma^ij*dj_K)
                      + beta^j*dj_Gamma^i - Gamma^j*dj_beta^i + (2/3)*Gamma^i*dj_beta^j
                      + gamma^jk*djk_beta^i + (1/3)*gamma^ij*djk_beta^k
        """
        from gr_solver.gr_core_fields import inv_sym6, sym6_to_mat33
        fields = solver.fields
        g_tilde = fields.gamma_tilde_sym6
        A_tilde = fields.A_sym6
        Gamma_tilde = fields.Gamma_tilde
        alpha = fields.alpha
        beta = fields.beta
        phi = fields.phi
        
        # Grid spacing (assuming uniform cubic)
        dx = fields.dx
        
        # Helper for central difference
        def d_i(f, axis):
            return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2 * dx)
            
        def d_ij(f, ax1, ax2):
            # d_i (d_j f)
            return d_i(d_i(f, ax2), ax1)

        # 1. Inverse metric and Christoffels
        g_inv_sym = inv_sym6(g_tilde)
        g_inv = sym6_to_mat33(g_inv_sym) # (..., 3, 3)
        g_mat = sym6_to_mat33(g_tilde)
        
        # Compute conformal Christoffel symbols Gamma^i_jk
        # Gamma^i_jk = 0.5 * g^il * (d_j g_lk + d_k g_lj - d_l g_jk)
        dg = np.zeros(g_mat.shape + (3,)) # (..., 3, 3, 3) last is deriv index
        for k in range(3):
            dg[..., k] = d_i(g_mat, k)
            
        Gamma_ijk = np.zeros(g_mat.shape + (3,)) # (..., 3, 3, 3) i, j, k
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    term = 0.0
                    for l in range(3):
                        term += 0.5 * g_inv[..., i, l] * (dg[..., l, k, j] + dg[..., l, j, k] - dg[..., j, k, l])
                    Gamma_ijk[..., i, j, k] = term

        # 2. Raise A_tilde: A^ij = g^ik g^jl A_kl
        A_mat = sym6_to_mat33(A_tilde)
        A_up = np.einsum('...ik,...jl,...kl->...ij', g_inv, g_inv, A_mat)
        
        # 3. Derivatives
        d_alpha = np.stack([d_i(alpha, k) for k in range(3)], axis=-1)
        d_phi = np.stack([d_i(phi, k) for k in range(3)], axis=-1)
        
        # K is trace of extrinsic curvature. Here we approximate K=0 or compute from K_sym6 if available.
        # In this test setup, K_sym6 is set. K = tr(K_ij) / e^4phi approx? 
        # For MMS, let's assume K=0 as phi=0 and A is traceless part of K.
        d_K = np.zeros_like(d_phi) 

        # 4. Advection terms (Lie derivative part 1)
        # beta^j dj_Gamma^i
        advect = np.zeros_like(Gamma_tilde)
        for j in range(3):
            advect += beta[..., j:j+1] * d_i(Gamma_tilde, j)
            
        # - Gamma^j dj_beta^i
        d_beta = np.zeros(beta.shape + (3,)) # (..., i, j) -> dj beta^i
        for j in range(3):
            d_beta[..., j] = d_i(beta, j) # deriv index is last
            
        twist = np.einsum('...j,...ji->...i', Gamma_tilde, d_beta)
        
        # (2/3) Gamma^i dj_beta^j
        div_beta = d_beta[..., 0, 0] + d_beta[..., 1, 1] + d_beta[..., 2, 2]
        compress = (2.0/3.0) * Gamma_tilde * div_beta[..., None]

        # 5. Second derivative terms of beta
        dd_beta = np.zeros(beta.shape + (3, 3)) # (..., i, j, k) -> dj dk beta^i
        for j in range(3):
            for k in range(3):
                dd_beta[..., j, k] = d_ij(beta, j, k)
                
        lap_shift = np.einsum('...jk,...ijk->...i', g_inv, dd_beta)
        grad_div_shift = np.einsum('...ij,...kjk->...i', g_inv, dd_beta)

        # 6. Assemble RHS
        rhs = np.zeros_like(Gamma_tilde)
        
        # Term: -2 A^ij dj_alpha
        rhs += -2.0 * np.einsum('...ij,...j->...i', A_up, d_alpha)
        
        # Term: 2 alpha ( Gamma^i_jk A^jk + 6 A^ij dj_phi - (2/3) g^ij dj_K )
        term_paren = np.einsum('...ijk,...jk->...i', Gamma_ijk, A_up)
        term_paren += 6.0 * np.einsum('...ij,...j->...i', A_up, d_phi)
        term_paren -= (2.0/3.0) * np.einsum('...ij,...j->...i', g_inv, d_K)
        rhs += 2.0 * alpha[..., None] * term_paren
        
        # Add Lie terms
        rhs += advect - twist + compress
        
        # Add shift Laplacian terms
        rhs += lap_shift + (1.0/3.0) * grad_div_shift
        
        return rhs

    def run(self):
        errors = []
        L = 16.0
        for N in [16, 32, 64]:
            dx = L / N
            dy = dx
            dz = dx
            solver = GRSolver(N, N, N, dx=dx, dy=dx, dz=dx)
            gamma, K, alpha, beta = self.set_mms(0, N, dx, dy, dz, L)
            solver.fields.gamma_sym6 = gamma
            solver.fields.K_sym6 = K
            solver.fields.alpha = alpha
            solver.fields.beta = beta
            solver.fields.phi = np.zeros((N, N, N))
            solver.fields.gamma_tilde_sym6 = gamma.copy()
            solver.fields.A_sym6 = K.copy()
            solver.fields.Gamma_tilde = self.compute_Gamma_tilde_mms(0, N, dx, dy, dz, L)
            solver.fields.Z = np.zeros((N, N, N))
            solver.fields.Z_i = np.zeros((N, N, N, 3))
            solver.geometry.compute_christoffels()
            solver.geometry.compute_ricci()
            solver.geometry.compute_scalar_curvature()
            dt_gamma, dt_K, dt_alpha, dt_beta = self.compute_dt_mms(0, N, dx, dy, dz, L)
            solver.stepper.compute_rhs()
            # Compute initial constraints
            solver.constraints.compute_hamiltonian()
            solver.constraints.compute_momentum()
            solver.constraints.compute_residuals()
            eps_H_init = solver.constraints.eps_H
            eps_M_init = solver.constraints.eps_M
            logging.debug(f"Test1 MMS N={N}: initial eps_H={eps_H_init}, eps_M={eps_M_init}")
            cfl = 0.01
            cfl_ratio = cfl
            logging.debug(f"Test1 MMS N={N}: CFL ratio dt/dx = {cfl_ratio}")
            S_gamma = dt_gamma - solver.stepper.rhs_gamma_sym6
            S_K = dt_K - solver.stepper.rhs_K_sym6
            S_phi = -solver.stepper.rhs_phi
            S_gamma_tilde = dt_gamma - solver.stepper.rhs_gamma_tilde_sym6
            S_A = dt_K - solver.stepper.rhs_A_sym6
            
            # Compute S_Gamma_tilde
            dt_Gamma_tilde = self.compute_dt_Gamma_tilde_mms(0, N, dx, dy, dz, L)
            rhs_Gamma_full_init = self.compute_full_gamma_driver_rhs(solver)
            S_Gamma_tilde = dt_Gamma_tilde - rhs_Gamma_full_init
            
            S_Z = -solver.stepper.rhs_Z
            S_Z_i = -solver.stepper.rhs_Z_i
            logging.debug(f"Test1 MMS N={N}: S_gamma_norm = {np.linalg.norm(S_gamma)}, S_K_norm = {np.linalg.norm(S_K)}, S_phi_norm = {np.linalg.norm(S_phi)}, S_gamma_tilde_norm = {np.linalg.norm(S_gamma_tilde)}, S_A_norm = {np.linalg.norm(S_A)}")
            solver.stepper.S_gamma_sym6 = S_gamma
            solver.stepper.S_K_sym6 = S_K
            solver.stepper.S_phi = S_phi
            solver.stepper.S_gamma_tilde_sym6 = S_gamma_tilde
            solver.stepper.S_A_sym6 = S_A
            solver.stepper.S_Gamma_tilde = S_Gamma_tilde
            solver.stepper.S_Z = S_Z
            solver.stepper.S_Z_i = S_Z_i
            
            original_compute_rhs = solver.stepper.compute_rhs
            def compute_rhs_with_sources(stepper, t, slow_update):
                original_compute_rhs(t, slow_update)
                stepper.rhs_gamma_sym6 += stepper.S_gamma_sym6
                stepper.rhs_K_sym6 += stepper.S_K_sym6
                stepper.rhs_phi += stepper.S_phi
                stepper.rhs_gamma_tilde_sym6 += stepper.S_gamma_tilde_sym6
                stepper.rhs_A_sym6 += stepper.S_A_sym6
                stepper.rhs_Z += stepper.S_Z
                stepper.rhs_Z_i += stepper.S_Z_i
                
                # Patch: Compute full Gamma_tilde evolution
                rhs_Gamma_full = self.compute_full_gamma_driver_rhs(solver)
                # Apply source (S_Gamma is 0, so just set RHS)
                stepper.rhs_Gamma_tilde = rhs_Gamma_full + stepper.S_Gamma_tilde
                
            solver.stepper.compute_rhs = lambda t, slow_update: compute_rhs_with_sources(solver.stepper, t, slow_update)
            solver.stepper.damping_enabled = False
            # Hold gauge fixed
            solver.gauge.evolve_lapse = lambda dt: None
            solver.gauge.evolve_shift = lambda dt: None
            T = 0.001
            # Use adaptive dt
            solver.scheduler.compute_dt = solver.stepper.memory.suggest_dt
            solver.orchestrator.t = 0.0
            solver.orchestrator.step = 0
            while solver.orchestrator.t < T:
                dt_max = T - solver.orchestrator.t
                dt, _, _ = solver.orchestrator.run_step(dt_max)
            # Compute final constraints
            solver.constraints.compute_residuals()
            eps_H_final = solver.constraints.eps_H
            logging.debug(f"Test1 MMS N={N}: final eps_H={eps_H_final}")
            gamma_exact, K_exact, alpha_exact, beta_exact = self.set_mms(T, N, dx, dy, dz, L)
            err_gamma = solver.fields.gamma_sym6 - gamma_exact
            err_K = solver.fields.K_sym6 - K_exact
            error_gamma = np.linalg.norm(err_gamma) / np.sqrt(err_gamma.size)
            error_K = np.linalg.norm(err_K) / np.sqrt(err_K.size)
            error = error_gamma + error_K
            errors.append(error)
            logging.debug(f"Test1 MMS N={N}: error_gamma={error_gamma:.2e}, error_K={error_K:.2e}, total error={error:.2e}")
            # Log max sources
            max_S_gamma = np.max(np.abs(S_gamma_tilde))
            max_S_K = np.max(np.abs(S_K))
            logging.debug(f"Test1 MMS N={N}: max_S_gamma={max_S_gamma:.2e}, max_S_K={max_S_K:.2e}")
        e_N, e_2N, e_4N = errors
        if e_2N == 0:
            p_obs = np.inf
        else:
            p_obs = np.log2(e_N / e_2N)
        passed = p_obs > 1.5
        metrics = {'p_obs': p_obs, 'errors': [e_N, e_2N, e_4N]}
        diagnosis = f"Observed order p_obs = {p_obs:.2f}, {'passed' if passed else 'failed'} convergence check (>1.5)"
        return {'passed': passed, 'metrics': metrics, 'diagnosis': diagnosis}

if __name__ == "__main__":
    test = Test1MmsLite(None)
    result = test.run()
    print(result)