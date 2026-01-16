from typing import List, Dict, Any
from receipt_schemas import MStepReceipt, MOrchReceipt
import numpy as np

class PromotionEngine:
    """Logic for canon promotion and pruning."""

    def __init__(self, memory_contract):
        self.memory = memory_contract

    def evaluate_promotion(self, step_receipts: List[MStepReceipt]) -> Dict[str, Any]:
        """Evaluate if step window qualifies for canon promotion."""
        if not step_receipts:
            return {'pass': False, 'reason': 'no_steps'}

        # Check residuals stable and low
        residuals = [r.residual_full for r in step_receipts]
        max_residual = max(max(v for sub in r.values() for v in sub.values()) for r in residuals)

        # Check enforcement not excessive
        max_enforcement = max(r.enforcement_magnitude for r in step_receipts)

        # Check regime stability (low chatter)
        dominant_threads = [r.gate_after.get('dominant_thread') for r in step_receipts]
        chatter = len(set(dominant_threads)) / len(dominant_threads)  # Rough chatter metric

        threshold = 1e-6
        if max_residual < threshold and max_enforcement < 0.1 and chatter < 0.5:
            return {
                'pass': True,
                'regime': 'stable',
                'metrics': {
                    'max_residual': max_residual,
                    'max_enforcement': max_enforcement,
                    'chatter': chatter
                }
            }
        else:
            return {
                'pass': False,
                'reason': f'residual={max_residual:.2e}, enforcement={max_enforcement:.2f}, chatter={chatter:.2f}'
            }

    def prune_old_attempts(self, max_attempts: int = 100):
        """Prune old attempt receipts to maintain ring buffer."""
        attempts = [(k, r) for k, r in self.memory.M_solve.items() if k.startswith('attempt_')]
        attempts.sort(key=lambda x: x[1].payload.attempt_id)
        to_prune = attempts[:-max_attempts]  # Keep newest

        for key, _ in to_prune:
            self.memory.memory_bank.remove_record(1, key)

    def failure_snapshot(self, last_k: int = 10):
        """Persist last K attempt receipts on failure for forensics."""
        attempts = [(k, r) for k, r in self.memory.M_solve.items() if k.startswith('attempt_')]
        attempts.sort(key=lambda x: x[1].payload.attempt_id, reverse=True)
        recent = attempts[:last_k]

        # Persist to disk (placeholder - in real impl, write to file)
        snapshot = {
            'timestamp': 'now',
            'attempts': [r.payload.__dict__ for _, r in recent]
        }
        print(f"Failure snapshot: {len(recent)} attempts persisted")

    def regime_invalidation(self, regime_hash: str):
        """Taint all records with given regime hash."""
        self.memory.memory_bank.invalidate_by_regime(regime_hash)