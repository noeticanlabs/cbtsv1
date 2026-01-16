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

class GRGauge:
    def __init__(self, fields, geometry):
        self.fields = fields
        self.geometry = geometry

    def evolve_lapse(self, dt):
        """1+log slicing: \partial_t \alpha = -2 \alpha K."""
        from .gr_core_fields import inv_sym6, trace_sym6
        gamma_inv = inv_sym6(self.fields.gamma_sym6)
        K_trace = trace_sym6(self.fields.K_sym6, gamma_inv)
        self.fields.alpha += dt * (-2.0 * self.fields.alpha * K_trace)

    def evolve_shift(self, dt):
        """Gamma-driver shift: \partial_t \beta^i = (3/4) \alpha B^i, where B^i = \Gamma^i - \lambda^i for BSSN."""
        # Ensure Gamma is computed
        if not hasattr(self.geometry, 'Gamma') or self.geometry.Gamma is None:
            self.geometry.compute_christoffels()
        B = self.geometry.Gamma - self.fields.lambda_i  # (Nx, Ny, Nz, 3)
        self.fields.beta += dt * (0.75 * self.fields.alpha[:, :, :, np.newaxis] * B)

    def evolve_lambda(self, dt):
        """Evolve BSSN gauge variable \lambda_i: \partial_t \lambda_i = \partial_t \Gamma_i - 2/3 \partial_i trK + ... Simplified."""
        # For simplicity, \partial_t \lambda_i = 0 (frozen) or link to Gamma
        pass  # Not implemented yet

    def compute_gradients(self):
        """Compute gradients of alpha and beta using finite differences."""
        dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz
        Nx, Ny, Nz = self.fields.Nx, self.fields.Ny, self.fields.Nz

        # Gradient of alpha
        grad_alpha = np.zeros((Nx, Ny, Nz, 3))
        grad_alpha[..., 0] = np.gradient(self.fields.alpha, dx, axis=0)
        grad_alpha[..., 1] = np.gradient(self.fields.alpha, dy, axis=1)
        grad_alpha[..., 2] = np.gradient(self.fields.alpha, dz, axis=2)

        # Gradient of beta (vector field)
        grad_beta = np.zeros((Nx, Ny, Nz, 3, 3))  # grad_beta[i][j,k] is ∂_i β^j
        for comp in range(3):
            grad_beta[..., comp, 0] = np.gradient(self.fields.beta[..., comp], dx, axis=0)
            grad_beta[..., comp, 1] = np.gradient(self.fields.beta[..., comp], dy, axis=1)
            grad_beta[..., comp, 2] = np.gradient(self.fields.beta[..., comp], dz, axis=2)

        return grad_alpha, grad_beta

    def compute_dt_gauge(self):
        """Compute dt_gauge = min(1/sqrt(grad α), 1/max(|grad β|))."""
        grad_alpha, grad_beta = self.compute_gradients()

        # Magnitude of grad alpha
        grad_alpha_mag = np.sqrt(np.sum(grad_alpha**2, axis=-1))

        # Magnitude of each component's gradient, then max over components
        grad_beta_mag = np.sqrt(np.sum(grad_beta**2, axis=-2))  # sum over derivative indices, leaving (Nx,Ny,Nz,3)
        max_grad_beta = np.max(grad_beta_mag, axis=-1)  # max over components

        # Avoid division by zero
        dt_alpha = 1.0 / (np.sqrt(grad_alpha_mag) + 1e-15)
        dt_beta = 1.0 / (max_grad_beta + 1e-15)

        dt_gauge = np.minimum(dt_alpha, dt_beta).min()
        return dt_gauge