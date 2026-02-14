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

from .gr_clock import UnifiedClock, UnifiedClockState


@dataclass
class TimeState:
    """Typed time objects for LoC-Time management."""
    t: float = 0.0  # Physical time
    n: int = 0      # Step number
    tau: float = 0.0  # Coherence time
    stage_times: Optional[Dict[str, float]] = field(default_factory=dict)  # Per-stage or per-clock times for Level 5+


class GRScheduler:
    """
    Scheduler for GR timestep selection with unified clock support.
    
    The scheduler can use either:
    1. A UnifiedClock for shared time state (recommended)
    2. Internal state for backward compatibility
    
    When using UnifiedClock, dt constraint computation is delegated to
    the clock's compute_dt_constraints() method, ensuring single source
    of truth for time state.
    """
    
    def __init__(self, fields, c=1.0, Lambda=0.0, rho_target=0.8,
                 unified_clock: Optional[UnifiedClock] = None):
        """
        Initialize the scheduler.
        
        Args:
            fields: GR fields object
            c: Speed of light (default: 1.0)
            Lambda: Cosmological constant (default: 0.0)
            rho_target: Target density for physical timestep (default: 0.8)
            unified_clock: Optional UnifiedClock for shared time state
        """
        self.fields = fields
        self.c = c
        self.Lambda = Lambda
        self.rho_target = rho_target
        self.max_dt = 0.1
        self.fixed_dt = None
        self.time_state = TimeState()  # Initialize time objects
        
        # Unified clock for shared state (optional for backward compatibility)
        self._unified_clock = unified_clock
        if unified_clock is not None:
            self._use_unified = True
        else:
            self._use_unified = False
        
        # State synchronization tracking
        self._last_synced_step = -1
        self._last_synced_time = -1.0
    
    @property
    def unified_clock(self) -> Optional[UnifiedClock]:
        """Get the unified clock if available."""
        return self._unified_clock
    
    def set_unified_clock(self, clock: UnifiedClock):
        """Set the unified clock for shared time state."""
        self._unified_clock = clock
        self._use_unified = clock is not None
    
    def _verify_clock_consistency(self):
        """
        Verify scheduler clock state matches unified clock.
        
        This synchronization point ensures that the scheduler and unified clock
        remain coherent throughout evolution. Called before major operations
        to detect desynchronization early.
        
        Raises:
            AssertionError: If clock state is inconsistent or missing
        """
        if self._unified_clock is None:
            return  # No unified clock, skip check
        
        # Verify unified clock state exists
        assert self._unified_clock.state is not None, "Clock state missing"
        
        # Check if global step has advanced (synchronization needed)
        current_step = self._unified_clock.state.global_step
        if current_step != self._last_synced_step:
            self._last_synced_step = current_step
            self._last_synced_time = self._unified_clock.state.global_time
            # Update scheduler's time state if needed
            if self.time_state is not None:
                self.time_state.n = current_step
                self.time_state.t = self._last_synced_time
        
        # Verify fields object is initialized
        assert self.fields is not None, "Fields not initialized"
        
        # Additional coherence checks
        assert self._unified_clock.state.global_step >= 0, "Invalid global step"
        assert self._unified_clock.state.global_time >= 0, "Invalid global time"
    
    def compute_dt(self, eps_H, eps_M):
        """Aeonic dt = min(CFL, curv, constraint, gauge, Lambda, phys). Stub."""
        if self.fixed_dt is not None:
            return min(self.fixed_dt, self.max_dt)
        dx = self.fields.dx
        dt_cfl = 0.5 * dx / self.c  # Assuming hyperbolic waves
        K_norm = np.max(np.linalg.norm(self.fields.K_sym6, axis=-1))
        dt_curv = self.fields.dx / max(np.sqrt(K_norm), 1e-6)  # Curvature clock based on extrinsic curvature norm
        dt_constraint = 1.0 if eps_H < 1e-4 else 0.5
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

    def compute_clocks(self, dt_candidate, lambda_val=0.0):
        """
        Compute all clock constraints and choose dt.
        
        This method now delegates to UnifiedClock.compute_dt_constraints() when
        a unified clock is available, ensuring single source of truth for time state.
        
        For backward compatibility, if no unified clock is set, it uses the
        original internal computation logic.
        
        Args:
            dt_candidate: Proposed timestep
            lambda_val: Constraint damping coefficient
            
        Returns:
            Tuple of (clocks_dict, dt_used)
        """
        # Verify clock state consistency before computation
        self._verify_clock_consistency()
        
        # Use unified clock if available
        if self._use_unified and self._unified_clock is not None:
            return self._unified_clock.compute_dt_constraints(dt_candidate, self.fields, lambda_val)
        
        # Legacy computation for backward compatibility
        # CFL: dt < dx / c, where c is characteristic speed
        # Rough estimate: c ~ sqrt(alpha^2 + beta^2) for ADM
        c_max = np.sqrt(np.max(self.fields.alpha)**2 + np.max(np.linalg.norm(self.fields.beta, axis=-1))**2)
        h_min = min(self.fields.dx, self.fields.dy, self.fields.dz)
        dt_CFL = h_min / c_max if c_max > 0 else float('inf')

        # Gauge: dt < alpha * dx / |beta| or similar
        # Simplified: dt < alpha * h_min / (1 + |beta|)
        beta_norm = np.max(np.linalg.norm(self.fields.beta, axis=-1))
        dt_gauge = self.fields.alpha.max() * h_min / (1 + beta_norm)

        # Coherence (constraint damping): dt < 1 / lambda where lambda is damping rate
        dt_coh = 1.0 / max(lambda_val, 1e-6) if lambda_val > 0 else float('inf')

        # Resolution: dt < h_min / sqrt(K^2) or similar
        K_norm = np.max(np.linalg.norm(self.fields.K_sym6, axis=-1))
        dt_res = h_min / max(np.sqrt(K_norm), 1e-6)

        # Sigma (shock capturing or similar): placeholder
        dt_sigma = float('inf')  # Not implemented yet

        # Choose minimum
        dt_used = min(dt_candidate, dt_CFL, dt_gauge, dt_coh, dt_res, dt_sigma)

        clocks = {
            'dt_CFL': dt_CFL,
            'dt_gauge': dt_gauge,
            'dt_coh': dt_coh,
            'dt_res': dt_res,
            'dt_sigma': dt_sigma,
            'dt_used': dt_used
        }

        return clocks, dt_used
