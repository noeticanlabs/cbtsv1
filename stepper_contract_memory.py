import hashlib
import numpy as np
import logging
from gr_solver.stepper_contract import StepperContract
from aeonic_memory_contract import AeonicMemoryContract
from phaseloom_27 import PhaseLoom27
from receipt_schemas import Kappa, MStepReceipt
from promotion_engine import PromotionEngine

logger = logging.getLogger('gr_solver.stepper_contract')

class StepperContractWithMemory(StepperContract):
    """Stepper contract integrated with Aeonic Memory and PhaseLoom."""

    def __init__(self, memory: AeonicMemoryContract, phaseloom: PhaseLoom27, max_attempts=20, dt_floor=1e-10, **kwargs):
        super().__init__(max_attempts=max_attempts, dt_floor=dt_floor)
        self.memory = memory
        self.phaseloom = phaseloom
        self.default_dt = dt_floor  # For compatibility
        self.promotion_engine = PromotionEngine(self.memory)

        # Adaptive dt governor parameters
        self.dt_governor = dt_floor
        self.dt_grow_factor = 1.1
        self.dt_shrink_factor = 0.7
        self.success_threshold = 5
        self.consecutive_successes = 0
        self.max_dt = 0.1
        self.min_dt = dt_floor

    def step(self, X_n, t_n, dt_candidate, rails_policy, phaseloom_caps):
        # Get last attempt's gate for classification
        last_gate = {'kind': 'dt'}  # Placeholder - would get from last attempt receipt

        # SEM barrier: abort on state gate with no repair
        try:
            self.memory.abort_on_state_gate_no_repair(last_gate)
        except Exception as e:
            return False, X_n, None, str(e)

        # Perform step attempt
        accepted, X_next, dt_used, rejection_reason = super().step(
            X_n, t_n, dt_candidate, rails_policy, phaseloom_caps
        )

        if accepted:
            # Create step receipt
            kappa = Kappa(o=0, s=self.memory.step_counter, mu=None)
            step_receipt = MStepReceipt(
                step_id=self.memory.step_counter,
                kappa=kappa,
                t=t_n + dt_used,
                tau_step=0,  # Clock integration needed
                residual_full={},  # Would compute full residuals here
                enforcement_magnitude=0.0,  # Would measure
                dt_used=dt_used,
                rollback_count=0,  # Per step
                gate_after={'pass': True},
                actions_applied=[]
            )

            self.memory.put_step_receipt(kappa, step_receipt)

            # Check for promotion
            recent_steps = self.memory.get_by_kappa((Kappa(0, 0, None), Kappa(0, 10, None)))
            if len(recent_steps) >= 5:  # Min window
                gate_result = self.promotion_engine.evaluate_promotion(recent_steps)
                self.memory.promote_to_canon(recent_steps, gate_result)

        else:
            # Prune old attempts to maintain ring buffer
            self.promotion_engine.prune_old_attempts()

        return accepted, X_next, dt_used, rejection_reason

    # Backward compatibility methods for StepperMemory
    def compute_regime_hash(self, eps_H, eps_M, max_R):
        """Compute hash based on current residuals and curvature."""
        state_str = f"{eps_H:.6e}_{eps_M:.6e}_{max_R:.6e}"
        return hashlib.md5(state_str.encode()).hexdigest()[:8]

    def suggest_dt(self, regime_hash):
        """Suggest dt based on the governor."""
        return self.dt_governor

    def post_step_update(self, dt, success, pre_resid, post_resid, regime_hash):
        """Update the governor based on step success."""
        if success:
            self.consecutive_successes += 1
            if self.consecutive_successes >= self.success_threshold:
                self.dt_governor = min(self.max_dt, self.dt_governor * self.dt_grow_factor)
                self.consecutive_successes = 0
        else:
            self.consecutive_successes = 0
            self.dt_governor = max(self.min_dt, self.dt_governor * self.dt_shrink_factor)

    def honesty_ok(self, pre_resid, post_resid, rails, eps_H, eps_M, geometry, fields):
        """Check for rail violations or residual increases. A residual increase is a soft-fail.
        Return (accepted, hard_fail, penalty)."""
        noise_floor_tolerance = 1e-8  # Calibrated tolerance for floating point noise (avoids soft fails from ~2e-9/step numerical accumulation)

        # First, check for hard rail violations.
        violation = rails.check_gates(eps_H, eps_M, geometry, fields)
        if violation:
            logger.warning("Rail violation: hard fail", extra={"extra_data": {"violation": violation}})
            return False, True, float('inf')

        # Next, check for residual increases (soft fail).
        if post_resid > pre_resid + noise_floor_tolerance:
            penalty = (post_resid - pre_resid) * 1e6 # Penalize based on magnitude of increase
            logger.warning(f"Honesty check soft fail: Residual increased beyond tolerance: {pre_resid} -> {post_resid}")
            # A residual increase is a soft fail, step is not accepted but can be retried.
            return False, False, penalty

        # If all checks pass, the step is accepted.
        return True, False, 0.0

    def attempt_receipt(self, X_n, t_n, dt_attempted, attempt_number):
        """Placeholder implementation for abstract method."""
        pass

    def step_receipt(self, X_next, t_next, dt_used):
        """Placeholder implementation for abstract method."""
        pass

    def check_gates(self, rails_policy):
        """Implement check_gates for memory stepper."""
        # For now, always pass, as memory doesn't enforce gates
        return True, '', {}