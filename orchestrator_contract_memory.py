from gr_solver.orchestrator_contract import OrchestratorContract
from aeonic_memory_contract import AeonicMemoryContract
from receipt_schemas import MOrchReceipt

class OrchestratorContractWithMemory(OrchestratorContract):
    """Orchestrator contract integrated with Aeonic Memory."""

    def __init__(self, memory: AeonicMemoryContract, **kwargs):
        super().__init__(**kwargs)
        self.memory = memory

    def aggregate_window(self, accepted_step_receipts, window_def, perf_counters):
        # Only use accepted steps (M_step)
        if not accepted_step_receipts:
            return self.SEM_FAILURE, "no_accepted_steps"

        # Check minimum history before verification
        if not self.memory.validate_min_accepted_history(len(accepted_step_receipts)):
            return self.SEM_FAILURE, "insufficient_accepted_history"

        # Compute window aggregates
        max_residual = max(r.residual_full for r in accepted_step_receipts)  # Simplified
        regime = "stable" if max_residual < 1e-6 else "constraint_risk"

        # Create orch receipt
        orch_receipt = MOrchReceipt(
            o=window_def.get('o', 0),
            window_steps=[r.step_id for r in accepted_step_receipts],
            quantiles={'p50': max_residual * 0.5, 'p90': max_residual * 0.9, 'p99': max_residual},
            dominance_histogram={},  # Would compute
            chatter_score=0.0,  # Would compute
            regime_label=regime,
            promotions=[],  # Earned patterns
            verification_threshold=1e-6,
            verification_norm='L2',
            min_accepted_history=len(accepted_step_receipts),
            policy_hash=accepted_step_receipts[0].kappa.o  # Placeholder
        )

        self.memory.put_orch_receipt(orch_receipt)

        promotions_quarantines = []
        if regime == "stable":
            promotions_quarantines.append({'action': 'promote', 'target': 'verified_window'})

        return orch_receipt, promotions_quarantines, regime