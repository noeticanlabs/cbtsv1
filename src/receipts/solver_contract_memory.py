from src.contracts.solver_contract import SolverContract
from src.aeonic.aeonic_memory_contract import AeonicMemoryContract
from src.phaseloom.phaseloom_27 import PhaseLoom27
from src.receipts.receipt_schemas import Kappa, MSolveReceipt
import numpy as np
import hashlib

class SolverContractWithMemory(SolverContract):
    """Solver contract integrated with Aeonic Memory and PhaseLoom."""

    def __init__(self, memory: AeonicMemoryContract, phaseloom: PhaseLoom27, **kwargs):
        super().__init__(**kwargs)
        self.memory = memory
        self.phaseloom = phaseloom

    def compute_rhs(self, X, t, gauge_policy, sources_func):
        # SEM check: validate prerequisites
        if not self.memory.check_no_silent_zeros(X, None):  # geometry not passed, placeholder
            return self.SEM_FAILURE, "prerequisites_not_initialized"

        # Compute RHS as usual
        result = super().compute_rhs(X, t, gauge_policy, sources_func)
        if isinstance(result, tuple) and len(result) == 2:
            rhs, diagnostics = result

            # Compute proxy residuals at solve clock
            residuals = self.phaseloom.compute_residuals(X, None, gauge_policy)  # geometry placeholder

            # Validate SEM: no NaN/Inf
            if not self.phaseloom.validate_sem(residuals):
                return self.SEM_FAILURE, "nonfinite_residuals"

            # Arbitrate dt
            dt_cap, dominant_thread = self.phaseloom.arbitrate_dt(residuals)
            gate = self.phaseloom.get_gate_classification(dominant_thread, residuals.get(dominant_thread[:2], 0))

            # Create attempt receipt
            kappa = Kappa(o=0, s=0, mu=0)  # Placeholder kappa
            attempt_receipt = MSolveReceipt(
                attempt_id=self.memory.attempt_counter,
                kappa=kappa,
                t=t,
                tau_attempt=0,  # Clock integration needed
                residual_proxy={k: {scale: v} for (dom, scale), v in residuals.items()},
                dt_cap_min=dt_cap,
                dominant_thread=dominant_thread,
                actions=self.phaseloom.get_rails(dominant_thread),
                policy_hash=self.memory.compute_policy_hash(gauge_policy),
                state_hash=hashlib.sha256(b"state_placeholder").hexdigest(),
                stage_time=t,
                stage_id=0,
                sem_ok=True,
                rollback_count=0,
                perf={}
            )

            self.memory.put_attempt_receipt(kappa, attempt_receipt)

            return rhs, diagnostics
        return result