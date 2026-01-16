import hashlib
from gr_solver.stepper_contract import StepperContract
from aeonic_memory_contract import AeonicMemoryContract
from phaseloom_27 import PhaseLoom27
from receipt_schemas import Kappa, MStepReceipt
from promotion_engine import PromotionEngine

class StepperContractWithMemory(StepperContract):
    """Stepper contract integrated with Aeonic Memory and PhaseLoom."""

    def __init__(self, memory: AeonicMemoryContract, phaseloom: PhaseLoom27, max_attempts=20, dt_floor=1e-10, **kwargs):
        super().__init__(max_attempts=max_attempts, dt_floor=dt_floor)
        self.memory = memory
        self.phaseloom = phaseloom
        self.default_dt = dt_floor  # For compatibility
        self.promotion_engine = PromotionEngine(memory)

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
        """Suggest dt based on past stats for this regime."""
        # Placeholder: return default dt
        return self.dt_floor

    def post_step_update(self, dt, success, pre_resid, post_resid, regime_hash):
        """Store the step data."""
        # Placeholder
        pass

    def honesty_check(self, pre_resid, post_resid, rails, eps_H, eps_M, geometry, fields):
        """Check for rail violations or residual increases. Return violation string or None."""
        # Check residual increase
        if post_resid > pre_resid:
            return f"Residual increased: {pre_resid:.2e} -> {post_resid:.2e}"
        # Check rails
        violation = rails.check_gates(eps_H, eps_M, geometry, fields)
        if violation:
            return violation
        return None

    def attempt_receipt(self, X_n, t_n, dt_attempted, attempt_number):
        """Placeholder implementation for abstract method."""
        # Emit attempt receipt - placeholder
        pass

    def step_receipt(self, X_next, t_next, dt_used):
        """Placeholder implementation for abstract method."""
        # Emit step receipt - placeholder
        pass