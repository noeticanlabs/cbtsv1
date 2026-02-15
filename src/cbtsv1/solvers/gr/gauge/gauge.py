# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time"
]
LEXICON_SYMBOLS = {
    "\\alpha": "GR_gauge.lapse",
    "\\beta": "GR_gauge.shift"
}

import numpy as np
from config.gr_gauge_nsc import evolve_lapse_compiled, evolve_shift_compiled, compute_gradients_compiled, compute_dt_gauge_compiled
from ..geometry.core_fields import inv_sym6, trace_sym6

class GRGauge:
    def __init__(self, fields, geometry):
        self.fields = fields
        self.geometry = geometry

    def evolve_lapse(self, dt):
        """1+log slicing: \partial_t \alpha = -2 \alpha K using compiled function."""
        self.fields.alpha = evolve_lapse_compiled(self.fields.alpha, self.fields.gamma_sym6, self.fields.K_sym6, dt)

    def evolve_shift(self, dt):
        """Gamma-driver shift: \partial_t \beta^i = (3/4) \alpha B^i, where B^i = \Gamma^i - \lambda^i for BSSN using compiled function."""
        # Ensure Gamma is computed
        if not hasattr(self.geometry, 'Gamma') or self.geometry.Gamma is None:
            self.geometry.compute_christoffels()
        self.fields.beta = evolve_shift_compiled(self.fields.beta, self.fields.alpha, self.geometry.Gamma, self.fields.lambda_i, dt)

    def evolve_lambda(self, dt):
        """Evolve BSSN gauge variable \lambda_i: \partial_t \lambda_i = - (2/3) \partial_i K"""
        # Compute K = trK
        gamma_inv = inv_sym6(self.fields.gamma_sym6)
        K_trace = trace_sym6(self.fields.K_sym6, gamma_inv)
        
        # Compute gradients of K
        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz
        dK_x = np.gradient(K_trace, dx, axis=0)
        dK_y = np.gradient(K_trace, dy, axis=1)
        dK_z = np.gradient(K_trace, dz, axis=2)
        grad_K = np.array([dK_x, dK_y, dK_z]).transpose(1, 2, 3, 0)
        
        # RHS for lambda_i
        rhs_lambda_i = - (2.0 / 3.0) * grad_K
        
        # Evolve
        self.fields.lambda_i += dt * rhs_lambda_i

    def compute_gradients(self):
        """Compute gradients of alpha and beta using compiled function."""
        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz
        return compute_gradients_compiled(self.fields.alpha, self.fields.beta, dx, dy, dz)

    def compute_dt_gauge(self):
        """Compute dt_gauge = min(1/sqrt(grad α), 1/max(|grad β|)) using compiled function."""
        grad_alpha, grad_beta = self.compute_gradients()
        return compute_dt_gauge_compiled(grad_alpha, grad_beta)