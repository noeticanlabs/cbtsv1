from gr_solver.orchestrator_contract import OrchestratorContract
from aeonic_memory_contract import AeonicMemoryContract
from receipt_schemas import MOrchReceipt
import numpy as np
import hashlib

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
        # Collect all residual values per domain
        all_residuals = {}
        for r in accepted_step_receipts:
            for domain, scales in r.residual_full.items():
                if domain not in all_residuals:
                    all_residuals[domain] = []
                all_residuals[domain].extend(scales.values())

        # Compute quantiles per domain
        quantiles = {}
        for domain, vals in all_residuals.items():
            quantiles[domain] = {
                'p50': np.percentile(vals, 50),
                'p90': np.percentile(vals, 90),
                'p99': np.percentile(vals, 99)
            }

        # Compute dominance histogram: count which domain is dominant (max residual) per step
        dominance_histogram = {}
        for r in accepted_step_receipts:
            max_val = 0
            dom = None
            for domain, scales in r.residual_full.items():
                val = max(scales.values())
                if val > max_val:
                    max_val = val
                    dom = domain
            if dom:
                dominance_histogram[dom] = dominance_histogram.get(dom, 0) + 1

        # Compute chatter_score as variance of max residuals per step
        max_residuals_per_step = [max(max(scales.values()) for scales in r.residual_full.values()) for r in accepted_step_receipts]
        chatter_score = np.var(max_residuals_per_step)

        # Overall max residual for regime
        max_residual = max(max_residuals_per_step) if max_residuals_per_step else 0
        regime = "stable" if max_residual < 1e-6 else "constraint_risk"

        # Create orch receipt
        orch_receipt = MOrchReceipt(
            o=window_def.get('o', 0),
            window_steps=[r.step_id for r in accepted_step_receipts],
            quantiles=quantiles,
            dominance_histogram=dominance_histogram,
            chatter_score=chatter_score,
            regime_label=regime,
            promotions=[],  # Earned patterns
            verification_threshold=1e-6,
            verification_norm='L2',
            min_accepted_history=len(accepted_step_receipts),
            policy_hash=hashlib.sha256(str(accepted_step_receipts[0].kappa.o).encode()).hexdigest()
        )

        self.memory.put_orch_receipt(orch_receipt)

        promotions_quarantines = []
        if regime == "stable":
            promotions_quarantines.append({'action': 'promote', 'target': 'verified_window'})

        return orch_receipt, promotions_quarantines, regime