# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time",
    "Aeonic_Phaseloom"
]
LEXICON_SYMBOLS = {
    "\\theta": "Aeonic_Phaseloom.phase",
    "\\rho": "Aeonic_Phaseloom.amplitude",
    "\\omega": "Aeonic_Phaseloom.rate"
}

import numpy as np

class GRPhaseLoomAdapter:
    def __init__(self, Nx, Ny, Nz):
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        # 3x3x3 probe grid: pick indices dividing domain
        self.probe_indices = []
        dx = Nx // 4
        dy = Ny // 4
        dz = Nz // 4
        for ix in [dx, 2*dx, 3*dx]:
            for iy in [dy, 2*dy, 3*dy]:
                for iz in [dz, 2*dz, 3*dz]:
                    self.probe_indices.append((ix, iy, iz))
        assert len(self.probe_indices) == 27

        # Thread tiers: 9 constraint, 9 gauge, 9 geometry
        self.N_threads = 27
        # History for phase unwrapping and rates
        self.theta_prev = np.zeros(self.N_threads)
        self.t_prev = 0.0
        self.initialized = False

    def extract_thread_signals(self, fields, constraints, geometry):
        """Extract (a_i, b_i) pairs for each of 27 threads."""
        a_vals = np.zeros(self.N_threads)
        b_vals = np.zeros(self.N_threads)

        for k, (ix, iy, iz) in enumerate(self.probe_indices):
            # Tier 0-8: Constraint threads
            if k < 9:
                a_vals[k] = constraints.H[ix, iy, iz]
                b_vals[k] = constraints.M[ix, iy, iz, 0]  # First component of M vector
            # Tier 9-17: Gauge threads
            elif k < 18:
                a_vals[k] = fields.alpha[ix, iy, iz] - 1.0  # Deviation from 1
                b_vals[k] = fields.beta[ix, iy, iz, 0]  # First shift component
            # Tier 18-26: Geometry threads
            else:
                # Proxy for geometry: deviation in gamma_xx and K_xx
                a_vals[k] = fields.gamma_sym6[ix, iy, iz, 0] - 1.0  # gamma_xx -1
                b_vals[k] = fields.K_sym6[ix, iy, iz, 0]  # K_xx

        return a_vals, b_vals

    def compute_theta_rho_omega(self, a_vals, b_vals, t_current, dt):
        """Compute theta, rho, omega for all threads."""
        theta = np.arctan2(b_vals, a_vals)
        rho = np.sqrt(a_vals**2 + b_vals**2)

        # Unwrap phases
        if not self.initialized:
            self.theta_prev = theta.copy()
            self.t_prev = t_current
            self.initialized = True
            omega = np.zeros(self.N_threads)
        else:
            # Simple unwrap: if diff > pi, subtract 2pi
            theta_unwrapped = theta.copy()
            diff = theta - self.theta_prev
            theta_unwrapped -= 2 * np.pi * np.round(diff / (2 * np.pi))
            omega = (theta_unwrapped - self.theta_prev) / (t_current - self.t_prev)
            self.theta_prev = theta_unwrapped.copy()
            self.t_prev = t_current

        return theta, rho, omega