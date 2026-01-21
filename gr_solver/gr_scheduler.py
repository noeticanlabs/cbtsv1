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
from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class TimeState:
    """Typed time objects for LoC-Time management."""
    t: float = 0.0  # Physical time
    n: int = 0      # Step number
    tau: float = 0.0  # Coherence time
    stage_times: Optional[Dict[str, float]] = field(default_factory=dict)  # Per-stage or per-clock times for Level 5+

class GRScheduler:
    def __init__(self, fields, c=1.0, Lambda=0.0, rho_target=0.8):
        self.fields = fields
        self.c = c
        self.Lambda = Lambda
        self.rho_target = rho_target
        self.max_dt = 0.1
        self.fixed_dt = None
        self.time_state = TimeState()  # Initialize time objects

    def compute_dt(self, eps_H, eps_M):
        """Aeonic dt = min(CFL, curv, constraint, gauge, Lambda, phys). Stub."""
        if self.fixed_dt is not None:
            return min(self.fixed_dt, self.max_dt)
        dx = self.fields.dx
        dt_cfl = 0.5 * dx / self.c  # Assuming hyperbolic waves
        K_norm = np.max(np.linalg.norm(self.fields.K_sym6, axis=-1))
        dt_curv = self.fields.dx / max(np.sqrt(K_norm), 1e-6)  # Curvature clock based on extrinsic curvature norm
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

    def update_coherence_time(self, dt, margins, eps_UFE, eps_constraints, invariant_drift, weights=None, h_params=None):
        """Update coherence time tau with dilation formula using incoherence score R_n from residuals."""
        if h_params is None:
            h_params = {'h0': 0.5, 'm_sat': 0.1, 'h_max': 1.0, 'threshold': 1e-6}
        if weights is None:
            weights = {'ufe': 1.0, 'constraints': 1.0, 'drift': 1.0}

        # Compute incoherence score R_n from residuals (eps_UFE, eps_constraints, invariant_drift) and weights
        threshold = h_params.get('threshold', 1e-6)
        incoherence = (weights['ufe'] * eps_UFE +
                       weights['constraints'] * eps_constraints +
                       weights['drift'] * invariant_drift) / threshold
        R_n = max(0, 1 - incoherence)  # Incoherence score: high when coherent (R_n close to 1)

        # Worst margin m_*
        m_star = min(margins.values()) if margins else 0.0

        # Governor h(m_*) with dilation by R_n
        h0, m_sat, h_max = h_params['h0'], h_params['m_sat'], h_params['h_max']
        if m_star < 0:
            h = 0.0
        elif m_star <= m_sat:
            h = h0 * m_star / m_sat
        else:
            h = h_max

        # Dilate by incoherence: if low coherence (R_n low), slow tau advance
        h_dilated = h * R_n

        # Update tau
        delta_tau = h_dilated * dt
        self.time_state.tau += delta_tau

        return delta_tau, R_n

    def enforce_cct_invariants(self, margins, residuals, drifts, time_policy='level_3'):
        """Enforce CCT (Coherence Computation Time) invariants and determine failure modes."""
        invariants_ok = True
        failure_mode = None

        # Invariant 1: Margins must be non-negative (safety)
        m_star = min(margins.values()) if margins else 0.0
        if m_star < 0:
            invariants_ok = False
            failure_mode = 'rollback'  # Or 'halt' depending on policy

        # Invariant 2: Coherence not degraded excessively
        threshold = 1e-5
        if residuals.get('rms', 0) > threshold or abs(drifts.get('energy', 0)) > threshold:
            # Check if tau is advancing; if not, failure
            if self.time_state.tau == 0.0:  # Example: if no progress
                invariants_ok = False
                failure_mode = 'emergency_dt'

        # Invariant 3: Time policy consistency
        if time_policy == 'level_3' and self.time_state.tau < 0:
            invariants_ok = False
            failure_mode = 'halt_integrator'

        # Failure modes: rollback, emergency_dt, halt_integrator, freeze_tau
        if not invariants_ok and failure_mode is None:
            failure_mode = 'freeze_tau'  # Default

        return invariants_ok, failure_mode