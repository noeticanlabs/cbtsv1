# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time",
    "Aeonic_Phaseloom"
]
LEXICON_SYMBOLS = {
    "\\Delta t_{\\text{loom}}": "Aeonic_Phaseloom.dt_control",
    "\\mu_{\\text{scale}}": "Aeonic_Phaseloom.damping_scale"
}

import numpy as np

class GRPhaseLoomController:
    def __init__(self, O_max=8, dt_base=0.1, C_loom=0.25, C_min=0.05, D_min=0.05, o_m=3, o_c=4, a_mu=0.25):
        self.O_max = O_max  # Max octaves
        self.dt_base = dt_base  # Base dt
        self.C_loom = C_loom  # Coherence threshold
        self.C_min = C_min
        self.D_min = D_min
        self.o_m = o_m  # Mid octave for coherence
        self.o_c = o_c  # Cutoff for tail danger
        self.a_mu = a_mu  # Scaling for mu update

    def compute_dt_loom(self, loom_data):
        """Compute dt_loom as min_j dt_band_j based on per-band activity."""
        D_max = loom_data['D_max']
        if D_max <= 1e-12:
            print("dt_loom: None (no danger)")
            return None
        omega_band = loom_data['omega_band']
        dt_band = np.zeros(self.O_max + 1)
        for o in range(self.O_max + 1):
            activity = np.mean(np.abs(omega_band[:, o])) + 1e-10
            dt_band[o] = self.C_loom / activity
            print(f"o={o}, activity={activity:.6e}, dt_band[o]={dt_band[o]:.6e}")
        dt_loom = np.min(dt_band)
        # Clip to reasonable range
        dt_loom = np.clip(dt_loom, 1e-6, 1.0)
        print(f"dt_loom: {dt_loom:.6e}")
        return dt_loom

    def compute_mu_scale(self, loom_data):
        """Compute damping scale from tail danger."""
        D_ge_oc = np.mean(loom_data['D_band'][self.o_c:])
        mu_scale = 1.0 + self.a_mu * D_ge_oc
        mu_scale = min(mu_scale, 10.0)  # Cap
        return mu_scale

    def get_controls(self, loom_data):
        """Get dt_loom and mu_scale."""
        dt_loom = self.compute_dt_loom(loom_data)
        mu_scale = self.compute_mu_scale(loom_data)
        return dt_loom, mu_scale