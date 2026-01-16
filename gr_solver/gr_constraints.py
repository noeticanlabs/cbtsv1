# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time"
]
LEXICON_SYMBOLS = {
    "\\mathcal{H}": "GR_constraint.hamiltonian",
    "\\mathcal{M}^i": "GR_constraint.momentum"
}

import numpy as np
import logging
from .logging_config import array_stats
from .gr_core_fields import inv_sym6, trace_sym6, norm2_sym6

logger = logging.getLogger('gr_solver.constraints')

class GRConstraints:
    def __init__(self, fields, geometry):
        self.fields = fields
        self.geometry = geometry

    def compute_hamiltonian(self):
        """Compute Hamiltonian constraint \mathcal{H}. Stub."""
        Nx, Ny, Nz = self.fields.Nx, self.fields.Ny, self.fields.Nz
        self.H = np.zeros((Nx, Ny, Nz))
        
        # Ensure geometry R is up to date
        if not hasattr(self.geometry, 'R') or self.geometry.R is None:
            self.geometry.compute_scalar_curvature()
            
        # R + K^2 - K_{ij} K^{ij}
        gamma_inv = inv_sym6(self.fields.gamma_sym6)
        K_trace = trace_sym6(self.fields.K_sym6, gamma_inv)
        K_sq = norm2_sym6(self.fields.K_sym6, gamma_inv)
        
        self.H = self.geometry.R + K_trace**2 - K_sq

    def compute_momentum(self):
        """Compute momentum constraints \mathcal{M}^i. Stub."""
        Nx, Ny, Nz = self.fields.Nx, self.fields.Ny, self.fields.Nz
        self.M = np.zeros((Nx, Ny, Nz, 3))
        # Gradient terms
        pass

    def compute_residuals(self):
        """Compute L2 norms \varepsilon_H, \varepsilon_M."""
        # L2 norm: sqrt( sum H^2 dV )
        dV = self.fields.dx * self.fields.dy * self.fields.dz
        self.eps_H = np.sqrt(np.sum(self.H**2) * dV)
        self.eps_M = np.sqrt(np.sum(self.M**2) * dV)  # M is 0 for now

        logger.debug("Computed constraint residuals", extra={
            "extra_data": {
                "eps_H": float(self.eps_H),
                "eps_M": float(self.eps_M),
                "H_stats": array_stats(self.H, "H"),
                "M_stats": array_stats(self.M, "M"),
                "dV": dV
            }
        })