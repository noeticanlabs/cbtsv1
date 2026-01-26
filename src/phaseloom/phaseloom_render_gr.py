# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time",
    "PhaseLoom"
]
LEXICON_SYMBOLS = {
    "H_channel": "PhaseLoom.render.constraint_H",
    "M_channel": "PhaseLoom.render.constraint_M",
    "R_channel": "PhaseLoom.render.curvature_R",
    "alpha_channel": "PhaseLoom.render.gauge_alpha",
    "beta_channel": "PhaseLoom.render.gauge_beta"
}

import numpy as np

class GRPhaseLoomRender:
    def __init__(self, fields, geometry, constraints):
        self.fields = fields
        self.geometry = geometry
        self.constraints = constraints
        self.channels = {
            'H': [],
            'M': [],
            'R': [],
            'alpha': [],
            'beta': []
        }

    def update_channels(self):
        """Extract scalars and update visualization channels"""
        # Constraint heatmaps
        H_map = self.constraints.H.copy()  # H(x)
        M_map = np.linalg.norm(self.constraints.M, axis=-1)  # |M|(x)

        # Curvature scalars
        R_map = self.geometry.R.copy()

        # Gauge fields
        alpha_map = self.fields.alpha.copy()
        beta_map = np.linalg.norm(self.fields.beta, axis=-1)  # |β|(x)

        # Append to channels (for timeline)
        self.channels['H'].append(H_map)
        self.channels['M'].append(M_map)
        self.channels['R'].append(R_map)
        self.channels['alpha'].append(alpha_map)
        self.channels['beta'].append(beta_map)

        # For viz, perhaps downsample or summarize
        # Here, just print max values for demo
        # print(f"Render: H_max={np.max(H_map):.2e}, M_max={np.max(M_map):.2e}, R_max={np.max(R_map):.2e}, alpha_min={np.min(alpha_map):.2e}, beta_max={np.max(beta_map):.2e}")

        # Clock strip: would be dt proposals, but for now, skip

    def compute_omega(self, prev_K=None, prev_gamma=None):
        """Compute spectral omega bands using compute_omega_current."""
        from .phaseloom_threads_gr import compute_omega_current
        return compute_omega_current(self.fields, prev_K, prev_gamma)