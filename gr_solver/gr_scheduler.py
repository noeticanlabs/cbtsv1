# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time"
]
LEXICON_SYMBOLS = {
    "\\Delta t": "CTL_time.step"
}

import numpy as np

class GRScheduler:
    def __init__(self, fields, c=1.0, Lambda=0.0, rho_target=0.8):
        self.fields = fields
        self.c = c
        self.Lambda = Lambda
        self.rho_target = rho_target
        self.max_dt = 0.1
        self.fixed_dt = None

    def compute_dt(self, eps_H, eps_M):
        """Aeonic dt = min(CFL, curv, constraint, gauge, Lambda, phys). Stub."""
        if self.fixed_dt is not None:
            return min(self.fixed_dt, self.max_dt)
        dx = self.fields.dx
        dt_cfl = 0.5 * dx / self.c  # Assuming hyperbolic waves
        dt_curv = 1.0  # Placeholder
        dt_constraint = 1.0 if eps_H < 1e-6 else 0.1
        dt_gauge = 1.0
        dt_lambda = np.sqrt(3 / abs(self.Lambda) / self.c**2) if self.Lambda != 0 else 1e10
        dt_phys = self.rho_target * dt_cfl  # dt_phys = rho_target * (C / v_max), with C=0.5, v_max=c
        self.dt = min(dt_cfl, dt_curv, dt_constraint, dt_gauge, dt_lambda, dt_phys, self.max_dt)
        return self.dt

    def compute_risk_gauge(self, proposals, dt, dt_loom=None):
        """Compute risk_gauge as min(margin) over active threads, including loom if active."""
        loom_active = dt_loom is not None and np.isfinite(dt_loom) and dt_loom > 0
        all_margins = []
        for k, p in proposals.items():
            p_dt = p.get('dt')
            if p_dt is None or not np.isfinite(p_dt) or p_dt <= 0:
                p['active'] = False
                p['ratio'] = 0.0
                p['margin'] = 1.0
            else:
                p['active'] = True
                p['ratio'] = dt / p_dt
                p['margin'] = 1 - p['ratio']
            if p['active']:
                all_margins.append(p['margin'])
            p['risk'] = p['margin']
        if loom_active:
            loom_margin = 1 - dt / dt_loom
            all_margins.append(loom_margin)
        risk_gauge = min(all_margins) if all_margins else 1.0
        return risk_gauge