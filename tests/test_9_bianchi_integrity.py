"""
Test 9 Bianchi Integrity
"""

import numpy as np
import logging
from gr_solver.gr_core_fields import inv_sym6, sym6_to_mat33, det_sym6

class Test9Bianchi:
    """
    Numerically verifies the contracted Bianchi identity div(G) = 0 
    (and thus div(T) = 0 via Einstein Eq) on a dynamical spacetime.
    """
    def __init__(self, gr_solver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        self.logger.info("Running Test9Bianchi (div T = 0 check)...")
        
        # 1. Initialize with a non-trivial wave (Minkowski + perturbation)
        self.gr_solver.init_minkowski()
        
        N = self.gr_solver.fields.Nx
        dx = self.gr_solver.fields.dx
        L = N * dx
        
        # Inject a smooth perturbation to generate curvature
        x = np.arange(N) * dx
        y = np.arange(N) * dx
        z = np.arange(N) * dx
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Perturbation in gamma_xx
        amp = 0.001
        k = 2 * np.pi / L
        pert = amp * np.sin(k * X) * np.sin(k * Y) * np.sin(k * Z)
        
        # Apply to physical gamma
        self.gr_solver.fields.gamma_sym6[..., 0] += pert
        
        # Enforce BSSN algebraic constraints (phi, gamma_tilde)
        det_g = det_sym6(self.gr_solver.fields.gamma_sym6)
        self.gr_solver.fields.phi = (1.0/12.0) * np.log(det_g)
        self.gr_solver.fields.gamma_tilde_sym6 = self.gr_solver.fields.gamma_sym6 * np.exp(-4.0 * self.gr_solver.fields.phi)[..., np.newaxis]
        
        # Perturbation in lapse to make g_00 interesting
        self.gr_solver.fields.alpha += 0.5 * pert
        
        # 2. Evolve and capture 7 time slices
        slices = []
        dt = 0.001 # Small dt for accuracy
        
        # Reset time
        self.gr_solver.orchestrator.t = 0.0
        self.gr_solver.orchestrator.step = 0
        
        for i in range(7):
            slices.append(self._capture_state())
            # Force a fixed small timestep for finite difference accuracy
            self.gr_solver.orchestrator.run_step(dt_max=dt)
            
        # 3. Compute Divergence at the central slice (index 3)
        G_prev, _ = self._compute_G_at(slices, 2, dt)
        G_curr, Gamma_curr = self._compute_G_at(slices, 3, dt)
        G_next, _ = self._compute_G_at(slices, 4, dt)
        
        # Derivatives of G^mu_nu at center (index 3)
        dG_dt = (G_next - G_prev) / (2*dt)
        dG_dx = self._spatial_deriv(G_curr, 0)
        dG_dy = self._spatial_deriv(G_curr, 1)
        dG_dz = self._spatial_deriv(G_curr, 2)
        
        dG = np.zeros((4, 4, 4) + G_curr.shape[2:])
        dG[0] = dG_dt
        dG[1] = dG_dx
        dG[2] = dG_dy
        dG[3] = dG_dz
        
        # Divergence: nabla_mu G^mu_nu
        divG = np.zeros((4,) + G_curr.shape[2:])
        for nu in range(4):
            term = np.zeros_like(divG[0])
            for mu in range(4):
                term += dG[mu, mu, nu]
                for lam in range(4):
                    term += Gamma_curr[mu, mu, lam] * G_curr[lam, nu]
                for lam in range(4):
                    term += Gamma_curr[nu, mu, lam] * G_curr[mu, lam]
            divG[nu] = term
            
        # Check magnitude
        divG_norm = np.sqrt(np.mean(divG**2))
        threshold = 1e-3
        passed = divG_norm < threshold
        
        diagnosis = f"Bianchi residual {divG_norm:.2e} < {threshold}" if passed else f"Bianchi residual {divG_norm:.2e} >= {threshold}"
        return {'passed': passed, 'metrics': {'divG_norm': divG_norm}, 'diagnosis': diagnosis}

    def _capture_state(self):
        return {
            't': self.gr_solver.orchestrator.t,
            'gamma': self.gr_solver.fields.gamma_sym6.copy(),
            'alpha': self.gr_solver.fields.alpha.copy(),
            'beta': self.gr_solver.fields.beta.copy()
        }

    def _compute_G_at(self, slices, idx, dt):
        Gammas_local = []
        for j in [idx-1, idx, idx+1]:
            s_m = slices[j-1]
            s_c = slices[j]
            s_p = slices[j+1]
            g_loc = self._construct_4metric(s_c)
            dg_dt_loc = (self._construct_4metric(s_p) - self._construct_4metric(s_m)) / (2*dt)
            dg_dx_loc = self._spatial_deriv(g_loc, 0)
            dg_dy_loc = self._spatial_deriv(g_loc, 1)
            dg_dz_loc = self._spatial_deriv(g_loc, 2)
            dg_loc = np.zeros((4, 4, 4) + g_loc.shape[2:])
            dg_loc[0] = dg_dt_loc
            dg_loc[1] = dg_dx_loc
            dg_loc[2] = dg_dy_loc
            dg_loc[3] = dg_dz_loc
            g_inv_loc = self._inverse_4metric(g_loc)
            Gam_loc = np.zeros((4, 4, 4) + g_loc.shape[3:])
            for sig in range(4):
                for mu in range(4):
                    for nu in range(4):
                        val = 0.0
                        for lam in range(4):
                            val += 0.5 * g_inv_loc[sig, lam] * (dg_loc[mu, nu, lam] + dg_loc[nu, mu, lam] - dg_loc[lam, mu, nu])
                        Gam_loc[sig, mu, nu] = val
            Gammas_local.append(Gam_loc)
        
        Gamma_c = Gammas_local[1]
        dGamma_dt = (Gammas_local[2] - Gammas_local[0]) / (2*dt)
        dGamma_dx = self._spatial_deriv(Gamma_c, 0)
        dGamma_dy = self._spatial_deriv(Gamma_c, 1)
        dGamma_dz = self._spatial_deriv(Gamma_c, 2)
        dGamma = np.zeros((4, 4, 4, 4) + Gamma_c.shape[3:])
        dGamma[0] = dGamma_dt
        dGamma[1] = dGamma_dx
        dGamma[2] = dGamma_dy
        dGamma[3] = dGamma_dz
        
        R_tensor = np.zeros((4, 4, 4, 4) + Gamma_c.shape[3:])
        for rho in range(4):
            for sig in range(4):
                for mu in range(4):
                    for nu in range(4):
                        t1 = dGamma[mu, rho, nu, sig]
                        t2 = dGamma[nu, rho, mu, sig]
                        t3 = np.zeros_like(t1)
                        t4 = np.zeros_like(t1)
                        for lam in range(4):
                            t3 += Gamma_c[rho, mu, lam] * Gamma_c[lam, nu, sig]
                            t4 += Gamma_c[rho, nu, lam] * Gamma_c[lam, mu, sig]
                        R_tensor[rho, sig, mu, nu] = t1 - t2 + t3 - t4
        
        Ricci = np.zeros((4, 4) + Gamma_c.shape[3:])
        for sig in range(4):
            for nu in range(4):
                for rho in range(4):
                    Ricci[sig, nu] += R_tensor[rho, sig, rho, nu]
        
        g_c = self._construct_4metric(slices[idx])
        g_inv_c = self._inverse_4metric(g_c)
        R_scalar = np.zeros_like(Ricci[0,0])
        for sig in range(4):
            for nu in range(4):
                R_scalar += g_inv_c[sig, nu] * Ricci[sig, nu]
        
        G_tensor = np.zeros((4, 4) + Gamma_c.shape[3:])
        for mu in range(4):
            for nu in range(4):
                G_tensor[mu, nu] = Ricci[mu, nu] - 0.5 * R_scalar * g_c[mu, nu]
        
        G_up = np.zeros_like(G_tensor)
        for mu in range(4):
            for nu in range(4):
                term = np.zeros_like(G_tensor[0,0])
                for alpha in range(4):
                    for beta in range(4):
                        term += g_inv_c[mu, alpha] * g_inv_c[nu, beta] * G_tensor[alpha, beta]
                G_up[mu, nu] = term
        return G_up, Gamma_c

    def _construct_4metric(self, state):
        gamma_sym = state['gamma']
        alpha = state['alpha']
        beta = state['beta']
        gamma = sym6_to_mat33(gamma_sym)
        beta_lower = np.einsum('...ij,...j->...i', gamma, beta)
        beta_sq = np.einsum('...i,...i->...', beta, beta_lower)
        g4 = np.zeros((4, 4) + alpha.shape)
        g4[0, 0] = -alpha**2 + beta_sq
        for i in range(3):
            g4[0, i+1] = beta_lower[..., i]
            g4[i+1, 0] = beta_lower[..., i]
            for j in range(3):
                g4[i+1, j+1] = gamma[..., i, j]
        return g4

    def _inverse_4metric(self, g4):
        g4_t = np.moveaxis(g4, [0, 1], [-2, -1])
        g4_inv_t = np.linalg.inv(g4_t)
        g4_inv = np.moveaxis(g4_inv_t, [-2, -1], [0, 1])
        return g4_inv

    def _spatial_deriv(self, f, axis):
        dx = self.gr_solver.fields.dx
        return (np.roll(f, -1, axis=axis-3) - np.roll(f, 1, axis=axis-3)) / (2 * dx)