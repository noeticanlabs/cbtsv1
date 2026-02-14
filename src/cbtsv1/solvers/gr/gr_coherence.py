import warnings
import numpy as np
import logging

logger = logging.getLogger('gr_solver.coherence')

# Deprecation warning
warnings.warn(
    "gr_coherence.py is deprecated. Use coherence_integration.py for canonical coherence.",
    DeprecationWarning,
    stacklevel=2
)

class CoherenceOperator:
    def __init__(self, damping_enabled=True, lambda_val=0.0):
        self.damping_enabled = damping_enabled
        self.lambda_val = lambda_val

    def apply_projection(self, fields):
        """
        Apply projection to enforce constraints like det(gamma_tilde) = 1.
        Returns K_proj dict.
        """
        from .gr_core_fields import sym6_to_mat33, mat33_to_sym6
        K_proj = {
            'gamma_sym6': np.zeros_like(fields.gamma_sym6),
            'K_sym6': np.zeros_like(fields.K_sym6),
            'phi': np.zeros_like(fields.phi),
            'gamma_tilde_sym6': np.zeros_like(fields.gamma_tilde_sym6),
            'A_sym6': np.zeros_like(fields.A_sym6),
            'Gamma_tilde': np.zeros_like(fields.Gamma_tilde),
            'Z': np.zeros_like(fields.Z),
            'Z_i': np.zeros_like(fields.Z_i)
        }
        # Enforce det(gamma_tilde) = 1
        gamma_tilde_mat = sym6_to_mat33(fields.gamma_tilde_sym6)
        det = np.linalg.det(gamma_tilde_mat)
        det_correction = det ** (-1.0/3.0)
        gamma_tilde_new = gamma_tilde_mat * det_correction[..., np.newaxis, np.newaxis]
        gamma_tilde_proj_sym6 = mat33_to_sym6(gamma_tilde_new)
        K_proj['gamma_tilde_sym6'] = gamma_tilde_proj_sym6 - fields.gamma_tilde_sym6
        return K_proj

    def apply_boundary_conditions(self, fields):
        """
        Apply boundary conditions.
        For periodic, returns zero K_bc.
        Returns K_bc dict.
        """
        K_bc = {
            'gamma_sym6': np.zeros_like(fields.gamma_sym6),
            'K_sym6': np.zeros_like(fields.K_sym6),
            'phi': np.zeros_like(fields.phi),
            'gamma_tilde_sym6': np.zeros_like(fields.gamma_tilde_sym6),
            'A_sym6': np.zeros_like(fields.A_sym6),
            'Gamma_tilde': np.zeros_like(fields.Gamma_tilde),
            'Z': np.zeros_like(fields.Z),
            'Z_i': np.zeros_like(fields.Z_i)
        }
        return K_bc

    def compute_dominance(self, rhs_gamma_sym6, rhs_K_sym6, rhs_phi, rhs_gamma_tilde_sym6, rhs_A_sym6, rhs_Z, rhs_Z_i):
        B_norm = (np.linalg.norm(rhs_gamma_sym6) + np.linalg.norm(rhs_K_sym6) +
                  np.linalg.norm(rhs_phi) + np.linalg.norm(rhs_gamma_tilde_sym6) +
                  np.linalg.norm(rhs_A_sym6) + np.linalg.norm(rhs_Z) + np.linalg.norm(rhs_Z_i))
        K_norm = abs(self.lambda_val) * B_norm
        eps = 1e-10
        D_lambda = abs(self.lambda_val) * K_norm / (B_norm + eps)
        return D_lambda

    def apply_damping(self, fields):
        if not self.damping_enabled:
            logger.debug("Damping disabled")
            return
        decay_factor = np.exp(-self.lambda_val)
        old_max_K = np.max(np.abs(fields.K_sym6))
        fields.K_sym6 *= decay_factor
        new_max_K = np.max(np.abs(fields.K_sym6))
        logger.debug("Applied constraint damping", extra={
            "extra_data": {
                "lambda_val": self.lambda_val,
                "decay_factor": decay_factor,
                "K_max_before": old_max_K,
                "K_max_after": new_max_K
            }
        })