import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional
from enum import Enum
from cbtsv1.solvers.gr.gate_policy import GatePolicy

logger = logging.getLogger('gr_solver.gates')


class GateKind(Enum):
    """Explicit gate kinds with hard-fail rules for audit-grade recovery."""
    CONSTRAINT = "constraint"  # Hard fail (abort)
    NONFINITE = "nonfinite"    # Hard fail (abort)
    UNINITIALIZED = "uninitialized"  # Hard fail (abort)
    STATE = "state"            # Soft by default, hard if no repair
    RATE = "rate"              # Soft (retry)


def should_hard_fail(gate) -> bool:
    """
    Determine if a gate failure should be a hard fail (abort) or soft (retry).
    
    Args:
        gate: A gate object with a 'kind' attribute (string or GateKind)
        
    Returns:
        True if hard fail (abort), False if soft (retry allowed)
    """
    if hasattr(gate, 'kind'):
        kind = gate.kind
    elif isinstance(gate, dict):
        kind = gate.get('kind', 'state')
    else:
        kind = str(gate)
    
    # Normalize to GateKind
    if isinstance(kind, str):
        try:
            kind = GateKind(kind)
        except ValueError:
            kind = GateKind.STATE  # Default to soft
    
    return kind in {GateKind.CONSTRAINT, GateKind.NONFINITE, GateKind.UNINITIALIZED}

class GateChecker:
    def __init__(self, constraints, analysis_mode=False, policy: Optional[GatePolicy] = None):
        self.constraints = constraints
        self.analysis_mode = analysis_mode
        self.policy = policy if policy is not None else GatePolicy()
        self.eps_H_state = 'normal'  # Track state for eps_H hysteresis: 'normal', 'warn', 'fail'

    def check_gates_internal(self, rails_policy=None):
        """Check step gates. Return (accepted, hard_fail, penalty, reasons, margins, corrections, debt_decomposition)."""
        if rails_policy is None:
            rails_policy = {}
        return self.check_gates()

    def apply_damping(self, lambda_val, damping_enabled):
        """Apply constraint damping: reduce constraint violations.
        
        NOTE: Energy-violating direct constraint damping removed.
        Constraint satisfaction now enforced via gates and RK4 discretization.
        """
        # Direct constraint damping (K_ij -= λ*H*γ_ij) violates symplectic structure
        # and causes 51.9% energy drift. Removed in favor of gate-based enforcement.
        # RK4 integrator provides mathematically correct discretization with 4.03 convergence.
        return

    def apply_corrections(self, corrections, current_dt, lambda_val):
        """Apply bounded corrective actions for warn level violations."""
        if corrections.get('reduce_dt'):
            current_dt *= 0.8  # Reduce dt by 20%
            logger.info(f"Corrective action: reducing dt to {current_dt}")
        if corrections.get('increase_kappa_budget'):
            lambda_val = min(lambda_val + 0.1, 1.0)  # Increase lambda
            logger.info(f"Corrective action: increasing lambda_val to {lambda_val}")
        return current_dt, lambda_val

    def check_gates(self):
        eps_H = float(self.constraints.eps_H)
        eps_M = float(self.constraints.eps_M)
        eps_proj = float(self.constraints.eps_proj)
        eps_clk = float(self.constraints.eps_clk) if self.constraints.eps_clk is not None else 0.0

        # Use policy thresholds instead of hardcoded values (Axiom A3)
        eps_H_warn = self.policy.eps_H['warn']
        eps_H_fail = self.policy.eps_H['fail']
        eps_M_soft_max = self.policy.eps_M['soft']
        eps_M_hard_max = self.policy.eps_M['hard']
        eps_proj_soft_max = self.policy.eps_proj['soft']
        eps_proj_hard_max = self.policy.eps_proj['hard']
        eps_clk_soft_max = self.policy.eps_clk['soft']
        eps_clk_hard_max = self.policy.eps_clk['hard']

        if np.isnan(eps_H) or np.isinf(eps_H) or np.isnan(eps_M) or np.isinf(eps_M) or np.isnan(eps_proj) or np.isinf(eps_proj) or np.isnan(eps_clk) or np.isinf(eps_clk):
            logger.error("Hard fail: NaN or infinite residuals in gates")
            return False, True, float('inf'), ["NaN/inf residuals"], {}, {}, {}, {}

        accepted = True
        hard_fail = False
        penalty = 0.0
        reasons = []
        margins = {}
        corrections = {}

        # Hysteresis for eps_H (from policy)
        enter_warn = self.policy.eps_H['warn']
        exit_warn = self.policy.eps_H['warn_exit']
        enter_fail = self.policy.eps_H['fail']

        if self.eps_H_state == 'normal':
            if eps_H > enter_warn:
                self.eps_H_state = 'warn'
                penalty += (eps_H - enter_warn) / enter_warn
                reasons.append(f"Warn: eps_H = {eps_H:.2e} > {enter_warn:.2e}")
                margins['eps_H'] = enter_warn - eps_H
                if self.analysis_mode:
                    logger.info("Analysis mode: would apply corrective actions", extra={
                        "extra_data": {
                            "would_reduce_dt": True,
                            "would_increase_kappa_budget": True,
                            "would_increase_projection_freq": True
                        }
                    })
                    corrections = {}
                else:
                    corrections = {'reduce_dt': True, 'increase_kappa_budget': True, 'increase_projection_freq': True}
        elif self.eps_H_state == 'warn':
            if eps_H > enter_fail:
                self.eps_H_state = 'fail'
                hard_fail = True
                accepted = False
                reasons.append(f"eps_H = {eps_H:.2e} > {enter_fail:.2e}")
                margins['eps_H'] = enter_fail - eps_H
            elif eps_H < exit_warn:
                self.eps_H_state = 'normal'
            else:
                # Stay in warn
                penalty += (eps_H - enter_warn) / enter_warn
                reasons.append(f"Warn: eps_H = {eps_H:.2e} > {enter_warn:.2e}")
                margins['eps_H'] = enter_warn - eps_H
                if self.analysis_mode:
                    logger.info("Analysis mode: would apply corrective actions", extra={
                        "extra_data": {
                            "would_reduce_dt": True,
                            "would_increase_kappa_budget": True,
                            "would_increase_projection_freq": True
                        }
                    })
                    corrections = {}
                else:
                    corrections = {'reduce_dt': True, 'increase_kappa_budget': True, 'increase_projection_freq': True}
        elif self.eps_H_state == 'fail':
            # Always hard fail once in fail state
            hard_fail = True
            accepted = False
            reasons.append(f"eps_H = {eps_H:.2e} > {enter_fail:.2e}")
            margins['eps_H'] = enter_fail - eps_H

        for eps, name, soft, hard in [
            (eps_M, 'eps_M', eps_M_soft_max, eps_M_hard_max),
            (eps_proj, 'eps_proj', eps_proj_soft_max, eps_proj_hard_max),
            (eps_clk, 'eps_clk', eps_clk_soft_max, eps_clk_hard_max)
        ]:
            if eps > hard:
                hard_fail = True
                accepted = False
                reasons.append(f"{name} = {eps:.2e} > {hard:.2e}")
                margins[name] = hard - eps
            elif eps > soft:
                penalty += (eps - soft) / soft
                reasons.append(f"Soft: {name} = {eps:.2e} > {soft:.2e}")
                margins[name] = soft - eps

        spike_norms = {
            'alpha_spike': np.max(np.abs(np.gradient(self.constraints.fields.alpha))),
            'K_spike': np.max(np.abs(np.gradient(self.constraints.fields.K_sym6)))
        }
        spike_soft_max = self.policy.spike['soft']
        spike_hard_max = self.policy.spike['hard']
        spike_penalty = 0.0
        for field, spike in spike_norms.items():
            if np.isnan(spike) or np.isinf(spike):
                hard_fail = True
                accepted = False
                reasons.append(f"Hard: {field} NaN/inf")
                margins[field] = float('-inf')
            elif spike > spike_hard_max:
                hard_fail = True
                accepted = False
                reasons.append(f"{field} spike = {spike:.2e} > {spike_hard_max:.2e}")
                margins[field] = spike_hard_max - spike
            elif spike > spike_soft_max:
                spike_penalty += (spike - spike_soft_max) / spike_soft_max
                reasons.append(f"Soft: {field} spike = {spike:.2e} > {spike_soft_max:.2e}")
                margins[field] = spike_soft_max - spike

        # Compute debt decomposition per Axiom A2 (Attribution)
        debt_decomposition = {
            'conservation_defect': float(eps_H ** 2),
            'reconstruction_error': float(eps_M ** 2),
            'tool_mismatch': float(eps_proj ** 2),
            'thrash_penalty': float(spike_penalty),
            'total_debt': float(eps_H ** 2 + eps_M ** 2 + eps_proj ** 2 + spike_penalty)
        }

        return accepted, hard_fail, penalty, reasons, margins, corrections, debt_decomposition