# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time",
    "PhaseLoom"
]
LEXICON_SYMBOLS = {
    "\\Delta t": "CTL_time.step",
    "T_phys": "PhaseLoom.thread_phys",
    "T_curv": "PhaseLoom.thread_curv",
    "T_cons": "PhaseLoom.thread_cons",
    "T_gauge": "PhaseLoom.thread_gauge",
    "T_diff": "PhaseLoom.thread_diff",
    "T_det": "PhaseLoom.thread_det",
    "T_io": "PhaseLoom.thread_io"
}

import numpy as np

def dt_thread_phys(alpha, beta, gamma_inv_sym6, dx, C_cfl, rho_target=0.8, dt_selected=0.1):
    trace_gamma_inv = gamma_inv_sym6[..., 0] + gamma_inv_sym6[..., 3] + gamma_inv_sym6[..., 5]  # sum of diagonal
    v_char = np.abs(beta).sum(axis=-1) + alpha * np.sqrt(trace_gamma_inv)
    v_max = v_char.max()
    dt = C_cfl * dx / v_max if v_max > 0 else np.inf
    ratio = dt_selected / dt if dt > 0 and np.isfinite(dt) else 0.0
    margin = 1.0 - ratio
    return {
        "dt": dt,
        "ratio": ratio,
        "margin": margin,
        "dominant_metric": "v_char_max",
        "metrics": {"v_char_max": float(v_max)}
    }

def dt_thread_diffusion(mu_H, mu_M, dx, C_diff, dt_selected=0.1):
    mu_max = max(mu_H, mu_M)
    dt = C_diff * dx * dx / mu_max if mu_max > 0 else np.inf
    ratio = dt_selected / dt if dt > 0 and np.isfinite(dt) else 0.0
    margin = 1.0 - ratio
    return {
        "dt": dt,
        "ratio": ratio,
        "margin": margin,
        "dominant_metric": "mu_max",
        "metrics": {"mu_max": float(mu_max)}
    }

def dt_thread_gauge(gauge, dt_selected=0.1):
    dt = gauge.compute_dt_gauge()
    ratio = dt_selected / dt if dt > 0 and np.isfinite(dt) else 0.0
    margin = 1.0 - ratio
    return {
        "dt": dt,
        "ratio": ratio,
        "margin": margin,
        "dominant_metric": "dt_gauge",
        "metrics": {"dt_gauge": float(dt)}
    }

def dt_thread_curvature(R_scalar, K_sq, C_curv, dt_selected=0.1):
    if R_scalar is not None:
        stiff = np.sqrt(np.abs(R_scalar).max())
        label = "sqrt(|R|)"
    else:
        stiff = np.sqrt(K_sq.max())
        label = "sqrt(K_ij K^ij)"
    dt = C_curv / (stiff + 1e-15)
    ratio = dt_selected / dt if dt > 0 and np.isfinite(dt) else 0.0
    margin = 1.0 - ratio
    return {
        "dt": dt,
        "ratio": ratio,
        "margin": margin,
        "dominant_metric": label,
        "metrics": {label: float(stiff)}
    }

def dt_thread_constraints(eps_H, eps_M, eps_H_prev, eps_M_prev, dt_prev, eps_H_max, eps_M_max, C_cons, dt_selected=0.1):
    dot_H = (eps_H - eps_H_prev) / (dt_prev + 1e-15) if eps_H_prev is not None and dt_prev is not None else 0.0
    dot_M = (eps_M - eps_M_prev) / (dt_prev + 1e-15) if eps_M_prev is not None and dt_prev is not None else 0.0
    dt_H = np.inf if dot_H <= 0 else (eps_H_max - eps_H) / (dot_H + 1e-15)
    dt_M = np.inf if dot_M <= 0 else (eps_M_max - eps_M) / (dot_M + 1e-15)
    dt_raw = min(dt_H, dt_M)
    dt = C_cons * dt_raw
    ratio = dt_selected / dt if dt > 0 and np.isfinite(dt) else 0.0
    margin = 1.0 - ratio
    return {
        "dt": dt,
        "ratio": ratio,
        "margin": margin,
        "dominant_metric": "constraint_risk",
        "metrics": {"dot_eps_H": float(dot_H), "dot_eps_M": float(dot_M)}
    }

def dt_thread_determinant(det_gamma, det_gamma_prev, dt_prev, det_gamma_min, C_det, dt_selected=0.1):
    det_min = det_gamma.min()
    dot_det = (det_min - det_gamma_prev) / (dt_prev + 1e-15) if det_gamma_prev is not None and dt_prev is not None else 0.0
    if dot_det >= 0:
        dt = np.inf
    else:
        dt = C_det * (det_min - det_gamma_min) / (-dot_det + 1e-15)
    ratio = dt_selected / dt if dt > 0 and np.isfinite(dt) else 0.0
    margin = 1.0 - ratio
    return {
        "dt": dt,
        "ratio": ratio,
        "margin": margin,
        "dominant_metric": "det_min",
        "metrics": {"det_min": float(det_min), "dot_det": float(dot_det)}
    }

class GRPhaseLoomThreads:
    def __init__(self, fields, eps_H_target=1e-10, eps_M_target=1e-10, m_det_min=0.2, c=1.0, Lambda=0.0, mu_H=0.01, mu_M=0.01, rho_target=0.8):
        self.fields = fields
        self.eps_H_target = eps_H_target
        self.eps_M_target = eps_M_target
        self.m_det_min = m_det_min
        self.c = c
        self.Lambda = Lambda
        self.mu_H = mu_H
        self.mu_M = mu_M
        self.rho_target = rho_target

        # Safe defaults
        self.C_CFL = 0.5
        self.C_diff = 0.10
        self.eta_alpha = 0.05
        self.C_curv = 0.10
        self.C_cons = 0.8
        self.C_det = 0.8
        self.eps_H_max = 10 * eps_H_target
        self.eps_M_max = 10 * eps_M_target

    def propose_dts(self, eps_H, eps_M, m_det, eps_H_prev, eps_M_prev, m_det_prev, dt_prev, geometry, gauge, dt_selected):
        """Propose dt limits from each thread"""

        from .gr_core_fields import inv_sym6, trace_sym6, norm2_sym6
        dx = self.fields.dx
        gamma_inv_sym6 = inv_sym6(self.fields.gamma_sym6)
        K_trace = trace_sym6(self.fields.K_sym6, gamma_inv_sym6)

        proposals = {
            'phys': dt_thread_phys(self.fields.alpha, self.fields.beta, gamma_inv_sym6, dx, self.C_CFL, self.rho_target, dt_selected),
            'diff': dt_thread_diffusion(self.mu_H, self.mu_M, dx, self.C_diff, dt_selected),
            'gauge': dt_thread_gauge(gauge, dt_selected),
            'curv': dt_thread_curvature(geometry.R if hasattr(geometry, 'R') and geometry.R is not None else None, norm2_sym6(self.fields.K_sym6, gamma_inv_sym6), self.C_curv, dt_selected),
            'cons': dt_thread_constraints(eps_H, eps_M, eps_H_prev, eps_M_prev, dt_prev, self.eps_H_max, self.eps_M_max, self.C_cons, dt_selected),
            'det': dt_thread_determinant(m_det, m_det_prev, dt_prev, self.m_det_min, self.C_det, dt_selected),
            'io': {'dt': 10.0, 'ratio': 0.0, 'margin': 1.0, 'dominant_metric': 'io', 'metrics': {}}
        }

        return proposals

def compute_omega_current(fields, prev_K=None, prev_gamma=None, spectral_cache=None):
    """Compute spectral activity omega_current from combined field changes using FFT, binned into 3x3x3 k-space bins."""
    # Combined state delta: weighted changes in gamma and K
    w_gamma = 1.0
    w_K = 10.0  # Weight K changes more since they drive evolution

    # Gamma change (xx component)
    gamma_field = fields.gamma_sym6[..., 0]
    if prev_gamma is not None:
        change_gamma = gamma_field - prev_gamma
    else:
        change_gamma = gamma_field

    # K change (xx component)
    K_field = fields.K_sym6[..., 0]
    if prev_K is not None:
        change_K = K_field - prev_K
    else:
        change_K = K_field

    # Combined change
    change = w_gamma * change_gamma + w_K * change_K

    # Compute FFT
    freq = np.fft.rfftn(change)
    power = np.abs(freq)**2

    # Use precomputed bin maps
    if spectral_cache is not None:
        kx_bin = spectral_cache.kx_bin[:, None, None]
        ky_bin = spectral_cache.ky_bin[None, :, None]
        kz_bin = spectral_cache.kz_bin[None, None, :]
    else:
        # Fallback to old method (for compatibility)
        Nx, Ny, Nz = change.shape
        dx = fields.dx
        kx = np.fft.fftfreq(Nx, d=dx)
        ky = np.fft.fftfreq(Ny, d=dx)
        kz = np.fft.rfftfreq(Nz, d=dx)
        kx_mesh, ky_mesh, kz_mesh = np.meshgrid(kx, ky, kz, indexing='ij')
        kx_bins = np.linspace(kx.min(), kx.max(), 4)
        ky_bins = np.linspace(ky.min(), ky.max(), 4)
        kz_bins = np.linspace(kz.min(), kz.max(), 4)
        kx_bin = np.clip(np.digitize(kx_mesh, kx_bins) - 1, 0, 2)
        ky_bin = np.clip(np.digitize(ky_mesh, ky_bins) - 1, 0, 2)
        kz_bin = np.clip(np.digitize(kz_mesh, kz_bins) - 1, 0, 2)

    # Compute omega_current: sum power per 3x3x3 bin
    omega_current = np.zeros(27)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                idx = i * 9 + j * 3 + k
                mask = (kx_bin == i) & (ky_bin == j) & (kz_bin == k)
                if np.any(mask):
                    omega_current[idx] = np.sum(power[mask])
                else:
                    omega_current[idx] = 0.0

    return omega_current