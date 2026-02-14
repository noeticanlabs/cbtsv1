# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time"
]
LEXICON_SYMBOLS = {
    "SEM": "Semantic Enforcement Module",
    "Kappa": "Time Semantic Tag",
    "SEMFailure": "Semantic Hard Failure"
}

import numpy as np
import logging
from cbtsv1.framework.receipt_schemas import SEMFailure, Kappa
from .geometry.core_fields import det_sym6, eigenvalues_sym6

logger = logging.getLogger('gr_solver.sem')

class KappaValidator:
    """Validates time monotonicity and coherence time semantics."""

    def __init__(self, eps_H_soft=1e-6, eps_M_soft=1e-6, eps_H_hard=1e-3, eps_M_hard=1e-2):
        self.eps_H_soft = eps_H_soft
        self.eps_M_soft = eps_M_soft
        self.eps_H_hard = eps_H_hard
        self.eps_M_hard = eps_M_hard

    def validate_pre_step(self, eps_H, eps_M, t_current, kappa=None):
        """Pre-step validation: check residuals against LoC thresholds."""
        if not np.isfinite(eps_H) or not np.isfinite(eps_M):
            raise SEMFailure("SEM: Non-finite residuals detected")

        if eps_H > self.eps_H_hard or eps_M > self.eps_M_hard:
            raise SEMFailure(f"SEM: Hard LoC violation: eps_H={eps_H:.2e} > {self.eps_H_hard:.2e}, eps_M={eps_M:.2e} > {self.eps_M_hard:.2e}")

        if eps_H > self.eps_H_soft or eps_M > self.eps_M_soft:
            logger.warning("SEM: Soft LoC threshold exceeded", extra={
                "extra_data": {
                    "eps_H": eps_H,
                    "eps_M": eps_M,
                    "eps_H_soft": self.eps_H_soft,
                    "eps_M_soft": self.eps_M_soft
                }
            })

        # Validate kappa if provided
        if kappa and not isinstance(kappa, Kappa):
            raise SEMFailure("SEM: Invalid kappa type")

        return True

    def validate_post_step(self, eps_H, eps_M, t_current, t_prev, kappa=None):
        """Post-step validation: ensure time monotonicity and semantic coherence."""
        if t_current <= t_prev:
            raise SEMFailure(f"SEM: Time monotonicity violation: t_current={t_current} <= t_prev={t_prev}")

        # Validate kappa consistency
        if kappa:
            if kappa.s < 0:
                raise SEMFailure(f"SEM: Invalid kappa step index: {kappa.s}")

        # Re-check residuals
        self.validate_pre_step(eps_H, eps_M, t_current, kappa)

        return True

class SemanticBarriers:
    """Audit gates for semantic barriers in GR evolution."""

    def __init__(self, det_gamma_min=1e-12, causality_check_enabled=True):
        self.det_gamma_min = det_gamma_min
        self.causality_check_enabled = causality_check_enabled

    def check_det_gamma(self, gamma_sym6):
        """Check det(γ) > 0 barrier."""
        det_gamma = det_sym6(gamma_sym6)
        if np.any(det_gamma <= self.det_gamma_min):
            violating_indices = np.where(det_gamma <= self.det_gamma_min)
            logger.error("SEM: det(γ) barrier violation", extra={
                "extra_data": {
                    "det_gamma_min": det_gamma[violating_indices],
                    "threshold": self.det_gamma_min,
                    "indices": violating_indices
                }
            })
            raise SEMFailure(f"SEM: det(γ) <= {self.det_gamma_min} at indices {violating_indices}")

        return True

    def check_causality(self, alpha, beta, gamma_inv_sym6):
        """Check causality violations (superluminal propagation)."""
        if not self.causality_check_enabled:
            return True

        # Causality check: ensure the normal to the slice is timelike.
        # This requires alpha > 0.
        # A more stringent condition is that the 4-velocity of Eulerian observers is timelike,
        # which requires alpha^2 - beta_i beta^i > 0.
        # For simplicity, check that alpha >= 0 (timelike foliation)
        if np.any(alpha < 0):
            violating_indices = np.where(alpha < 0)
            logger.error("SEM: Causality violation - negative lapse", extra={
                "extra_data": {
                    "alpha_min": np.min(alpha),
                    "violating_indices": violating_indices
                }
            })
            raise SEMFailure(f"SEM: Causality violation - negative lapse at indices {violating_indices}")

        # More complete check: alpha^2 > beta_i beta^i
        if gamma_inv_sym6 is not None:
            from .gr_core_fields import sym6_to_mat33
            gamma_inv_full = sym6_to_mat33(gamma_inv_sym6)
            beta_u = np.einsum('...ij,...j->...i', gamma_inv_full, beta)
            beta_sq = np.einsum('...i,...i', beta, beta_u)
            if np.any(alpha**2 <= beta_sq):
                violating_indices = np.where(alpha**2 <= beta_sq)
                logger.error("SEM: Causality violation - superluminal shift", extra={
                    "extra_data": {
                        "violating_indices": violating_indices
                    }
                })
                raise SEMFailure(f"SEM: Causality violation - superluminal shift at indices {violating_indices}")

        return True

    def audit_step(self, fields, geometry):
        """Full semantic audit for a step."""
        from .gr_core_fields import inv_sym6
        self.check_det_gamma(fields.gamma_sym6)
        gamma_inv = inv_sym6(fields.gamma_sym6)
        self.check_causality(fields.alpha, fields.beta, gamma_inv)

        return True

class SEMDomain:
    """SEM domain integration with LoC axiom coherence thresholds."""

    def __init__(self, kappa_validator=None, barriers=None):
        self.kappa_validator = kappa_validator or KappaValidator()
        self.barriers = barriers or SemanticBarriers()

    def pre_step_check(self, eps_H, eps_M, t_current, kappa=None):
        """Pre-step SEM validation."""
        return self.kappa_validator.validate_pre_step(eps_H, eps_M, t_current, kappa)

    def post_step_audit(self, eps_H, eps_M, t_current, t_prev, fields, geometry, kappa=None):
        """Post-step SEM audit with rollback on failure."""
        try:
            self.kappa_validator.validate_post_step(eps_H, eps_M, t_current, t_prev, kappa)
            self.barriers.audit_step(fields, geometry)
            return True
        except SEMFailure as e:
            logger.error("SEM: Post-step audit failed", extra={
                "extra_data": {
                    "failure_reason": str(e),
                    "eps_H": eps_H,
                    "eps_M": eps_M,
                    "t_current": t_current,
                    "t_prev": t_prev
                }
            })
            raise  # Re-raise to trigger rollback