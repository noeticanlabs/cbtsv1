# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time",
    "PhaseLoom"
]
LEXICON_SYMBOLS = {
    "H_max": "PhaseLoom.rail_H_max",
    "M_max": "PhaseLoom.rail_M_max",
    "R_max": "PhaseLoom.rail_R_max"
}

import numpy as np
import json

class GRPhaseLoomRails:
    def __init__(self, fields, nsc_policy_path="gr_gate_policy.nsc", alpha_floor=1e-8, H_max=1e-4, M_max=1e-4, R_max=1e6, warning_threshold=0.8, lambda_floor=1e-8, kappa_max=1e12):
        self.fields = fields

        # Load NSC policy script and execute to get thresholds
        try:
            with open(nsc_policy_path, 'r') as f:
                nsc_code = f.read()
            global_vars = {'np': np, 'fields': fields}  # Provide numpy and fields for dynamic policies
            local_vars = {}
            exec(nsc_code, global_vars, local_vars)
            self.gate_thresholds = local_vars  # Use the executed variables as thresholds
        except FileNotFoundError:
            self.gate_thresholds = {}

        # Link to NSC glyph policies from gr_gates.py for dynamic gates
        from src.core.gr_gates import GateChecker
        # Assuming constraints are passed or available; for now, create a placeholder
        self.gate_checker = None  # Will need to set this with actual constraints

        # Override defaults with policy if available
        self.alpha_floor = self.gate_thresholds.get('alpha_floor', alpha_floor)
        self.H_max = self.gate_thresholds.get('H_max', H_max)
        self.H_warn = self.gate_thresholds.get('H_warn', 7.5e-5)
        self.M_max = self.gate_thresholds.get('M_max', M_max)
        self.R_max = self.gate_thresholds.get('R_max', R_max)
        self.warning_threshold = warning_threshold
        self.lambda_floor = self.gate_thresholds.get('lambda_floor', lambda_floor)
        self.kappa_max = self.gate_thresholds.get('kappa_max', kappa_max)

        # Incremental diagnostics cache
        self.prev_gamma = None
        self.cached_det_gamma = None
        self.cached_eigvals = None
        self.cached_kappa = None
        self.change_threshold = 1e-6  # Threshold for considering gamma unchanged

    def _get_gamma_diagnostics(self, fields):
        """Get gamma diagnostics, using cache if gamma unchanged."""
        gamma = fields.gamma_sym6
        if self.prev_gamma is not None:
            delta = np.max(np.abs(gamma - self.prev_gamma))
            if delta < self.change_threshold and self.cached_det_gamma is not None:
                return self.cached_det_gamma, self.cached_eigvals, self.cached_kappa

        # Compute fresh
        from src.core.gr_core_fields import det_sym6, eigenvalues_sym6, cond_sym6
        det_gamma = det_sym6(gamma)
        eigvals = eigenvalues_sym6(gamma)
        kappa = cond_sym6(gamma)

        # Cache
        self.prev_gamma = gamma.copy()
        self.cached_det_gamma = det_gamma
        self.cached_eigvals = eigvals
        self.cached_kappa = kappa

        return det_gamma, eigvals, kappa

    def check_gates(self, eps_H, eps_M, geometry, fields):
        """Check coherence gates: return violation description or None"""

        # If NSC defines a check_gates function, use it
        if 'check_gates' in self.gate_thresholds and callable(self.gate_thresholds['check_gates']):
            return self.gate_thresholds['check_gates'](eps_H, eps_M, geometry, fields, self)

        # Fallback to default checks
        # Hard fail gates
        if np.any(np.isnan(fields.gamma_sym6)) or np.any(np.isinf(fields.gamma_sym6)):
            return "NaN or inf in gamma"
        if np.any(np.isnan(fields.K_sym6)) or np.any(np.isinf(fields.K_sym6)):
            return "NaN or inf in K"
        if np.any(np.isnan(fields.alpha)) or np.any(np.isinf(fields.alpha)):
            return "NaN or inf in alpha"
        if np.any(np.isnan(fields.beta)) or np.any(np.isinf(fields.beta)):
            return "NaN or inf in beta"

        det_gamma, eigvals, kappa = self._get_gamma_diagnostics(fields)

        if np.any(det_gamma <= 0):
            return "det(gamma) <= 0"

        lambda_min = np.min(eigvals, axis=-1)
        if np.any(lambda_min <= self.lambda_floor):
            return f"lambda_min <= lambda_floor ({lambda_min.min():.2e} <= {self.lambda_floor:.2e})"

        if np.any(kappa >= self.kappa_max):
            return f"kappa >= kappa_max ({kappa.max():.2e} >= {self.kappa_max:.2e})"

        if np.any(fields.alpha <= self.alpha_floor):
            return "alpha <= alpha_floor"

        # Soft gates
        if eps_H > self.H_max:
            return f"fail: eps_H > H_max ({eps_H:.2e} > {self.H_max:.2e})"
        elif eps_H > self.H_warn:
            return f"warn: eps_H > H_warn ({eps_H:.2e} > {self.H_warn:.2e})"
        if eps_M > self.M_max:
            return f"eps_M > M_max ({eps_M:.2e} > {self.M_max:.2e})"

        max_R = np.max(geometry.R)
        if max_R > self.R_max:
            return f"R > R_max ({max_R:.2e} > {self.R_max:.2e})"

        # Drift gates (placeholder)
        # Asymmetry creep: check if gamma and K are still symmetric
        # For simplicity, assume they are, or add a check

        return None  # No violation

    def compute_margins(self, eps_H, eps_M, geometry, fields, m_det_min):
        """Compute rail margins as rail_trip / severity."""
        margins = {}
        # For upper limits: margin = current / threshold
        margins['eps_H'] = eps_H / self.H_max if self.H_max > 0 else 0.0
        margins['eps_M'] = eps_M / self.M_max if self.M_max > 0 else 0.0
        max_R = np.max(geometry.R) if hasattr(geometry, 'R') and geometry.R is not None else 0.0
        margins['R'] = max_R / self.R_max if self.R_max > 0 else 0.0
        # For lower limits: margin = threshold / current (inverted)
        det_gamma, eigvals, kappa = self._get_gamma_diagnostics(fields)

        det_min = det_gamma.min()
        margins['det'] = m_det_min / det_min if det_min > 0 else np.inf
        alpha_min = np.min(fields.alpha)
        margins['alpha'] = self.alpha_floor / alpha_min if alpha_min > 0 else np.inf

        lambda_min = np.min(eigvals, axis=-1)
        lambda_min_global = lambda_min.min()
        margins['lambda'] = self.lambda_floor / lambda_min_global if lambda_min_global > 0 else np.inf

        kappa_max_global = kappa.max()
        margins['kappa'] = kappa_max_global / self.kappa_max if self.kappa_max > 0 else 0.0

        return margins