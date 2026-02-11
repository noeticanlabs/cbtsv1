"""
Aeonic Memory Language (AML) - Legality Layer for NSC↔GR/NR Coupling

AML enforces the coupling policy v0.1 by providing a tagged execution environment
where operations are compartmentalized, memory is tiered, threads are tagged,
gates are enforced, and receipts are generated.
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
import json
import hashlib
import threading
import copy
from dataclasses import dataclass
from src.aeonic.aeonic_memory_contract import AeonicMemoryContract
from src.aeonic.aeonic_memory_bank import AeonicMemoryBank
from src.receipts.receipt_schemas import Kappa, MSolveReceipt, MStepReceipt, MOrchReceipt, SEMFailure
from src.aeonic.aeonic_receipts import AeonicReceipts
from src.aeonic.aeonic_clocks import AeonicClockPack
import numpy as np

@dataclass
class ThreadTag:
    """Thread tagging for compartment enforcement."""
    compartment: str  # SOLVE, STEP, ORCH, S_PHY
    operation: str    # e.g., "solve", "step", "enforce", "clamp"
    kappa: Optional[Kappa] = None
    policy_hash: Optional[str] = None

class AMLGate:
    """Gate protocol enforcement."""

    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds
        self.eps_H_max = thresholds.get('eps_H_max', 1e-4)
        self.eps_M_max = thresholds.get('eps_M_max', 1e-4)
        self.eps_SEM_hard = thresholds.get('eps_SEM_hard', 0.0)

    def check_gate(self, eps_H: float, eps_M: float) -> Tuple[bool, str]:
        """Check if residuals pass gate thresholds."""
        if eps_H > self.eps_H_max or eps_M > self.eps_M_max:
            return False, f"Residuals exceed thresholds: eps_H={eps_H:.2e} > {self.eps_H_max:.2e}, eps_M={eps_M:.2e} > {self.eps_M_max:.2e}"

        if eps_H == self.eps_SEM_hard or eps_M == self.eps_SEM_hard:
            return False, f"SEM hard failure: eps_H={eps_H:.2e}, eps_M={eps_M:.2e}"

        return True, "OK"

class AML:
    """Aeonic Memory Language - Legality Layer enforcing NSC↔GR/NR coupling."""

    def __init__(self, coupling_policy_path: str = "coupling_policy_v0.1.json"):
        # Load coupling policy
        with open(coupling_policy_path, 'r') as f:
            self.policy = json.load(f)

        self.compartments = self.policy['compartments']
        self.coupling_matrix = self.policy['coupling_matrix']
        self.gate_thresholds = self.policy['gate_thresholds']
        self.retry_schedules = self.policy['retry_schedules']
        self.ops_allowed_s_phy = self.policy['ops_allowed_to_touch_S_PHY']

        # Initialize components
        self.clock = AeonicClockPack()
        self.receipts = AeonicReceipts()
        self.memory_bank = AeonicMemoryBank(self.clock, self.receipts)
        self.memory_contract = AeonicMemoryContract(
            memory_bank=self.memory_bank,
            receipts_log=self.receipts
        )

        # Gate enforcement
        self.gate = AMLGate(self.gate_thresholds)

        # Thread-local storage for tags
        self.thread_local = threading.local()

        # Transaction state for commit/rollback
        self.transaction_stack: List[Dict[str, Any]] = []

    def tag_thread(self, compartment: str, operation: str, kappa: Optional[Kappa] = None) -> ThreadTag:
        """Tag the current thread with compartment and operation."""
        if compartment not in self.compartments:
            raise SEMFailure(f"Invalid compartment: {compartment}")

        tag = ThreadTag(compartment=compartment, operation=operation, kappa=kappa)
        self.thread_local.tag = tag
        return tag

    def get_current_tag(self) -> Optional[ThreadTag]:
        """Get the current thread's tag."""
        return getattr(self.thread_local, 'tag', None)

    def enforce_compartment_transition(self, from_compartment: str, to_compartment: str):
        """Enforce coupling matrix transitions."""
        if from_compartment not in self.coupling_matrix:
            raise SEMFailure(f"Unknown compartment: {from_compartment}")

        allowed = self.coupling_matrix[from_compartment]
        if to_compartment not in allowed:
            raise SEMFailure(f"Transition {from_compartment} -> {to_compartment} not allowed by coupling matrix")

    def protect_s_phy(self, operation: str):
        """Protect S_PHY compartment - only allowed operations."""
        tag = self.get_current_tag()
        if tag and tag.compartment == 'S_PHY':
            if operation not in self.ops_allowed_s_phy:
                raise SEMFailure(f"Operation '{operation}' not allowed in S_PHY compartment. Allowed: {self.ops_allowed_s_phy}")

    def begin_transaction(self, tag: ThreadTag):
        """Begin a transaction for commit/rollback."""
        transaction = {
            'tag': tag,
            'state': {
                'tiers': copy.deepcopy(self.memory_bank.tiers),
                'attempt_counter': self.memory_contract.attempt_counter,
                'step_counter': self.memory_contract.step_counter
            },
            'receipts': []
        }
        self.transaction_stack.append(transaction)

    def commit_transaction(self) -> bool:
        """Commit the current transaction."""
        if not self.transaction_stack:
            raise SEMFailure("No active transaction to commit")

        transaction = self.transaction_stack.pop()
        # Emit receipts
        for receipt in transaction['receipts']:
            if isinstance(receipt, MSolveReceipt):
                self.memory_contract.put_attempt_receipt(receipt.kappa, receipt)
            elif isinstance(receipt, MStepReceipt):
                self.memory_contract.put_step_receipt(receipt.kappa, receipt)
            elif isinstance(receipt, MOrchReceipt):
                self.memory_contract.put_orch_receipt(receipt)

        return True

    def rollback_transaction(self):
        """Rollback the current transaction."""
        if not self.transaction_stack:
            raise SEMFailure("No active transaction to rollback")

        transaction = self.transaction_stack.pop()
        # Restore state to ensure consistent state after rejection
        self.memory_bank.tiers = copy.deepcopy(transaction['state']['tiers'])
        self.memory_contract.attempt_counter = transaction['state']['attempt_counter']
        self.memory_contract.step_counter = transaction['state']['step_counter']

    def execute_operation(self, operation_func: Callable, *args, compartment: str, operation: str, kappa: Optional[Kappa] = None, **kwargs):
        """Execute an operation within AML legality layer."""
        # Tag thread
        tag = self.tag_thread(compartment, operation, kappa)

        # Begin transaction
        self.begin_transaction(tag)

        try:
            # Enforce compartment rules
            self.protect_s_phy(operation)

            # Execute the operation
            result = operation_func(*args, **kwargs)

            # Generate receipt based on compartment
            receipt = self.generate_receipt(tag, result)

            # Store receipt in transaction
            self.transaction_stack[-1]['receipts'].append(receipt)

            # Commit
            self.commit_transaction()

            return result

        except Exception as e:
            # Rollback on failure
            self.rollback_transaction()
            raise e

    def generate_receipt(self, tag: ThreadTag, result: Any) -> Any:
        """Generate appropriate receipt based on compartment and operation."""
        if tag.compartment == 'SOLVE':
            # MSolveReceipt for solve attempts
            return MSolveReceipt(
                attempt_id=self.memory_contract.attempt_counter + 1,
                kappa=tag.kappa,
                t=result.get('t', 0.0),
                tau_attempt=self.clock.tau_s,
                residual_proxy=result.get('residuals', {}),
                dt_cap_min=result.get('dt_min', 1e-6),
                dominant_thread=(tag.compartment, tag.operation),
                actions=result.get('actions', []),
                policy_hash=self.compute_policy_hash(self.policy),
                state_hash=self.hash_state(result.get('state', {})),
                stage_time=result.get('stage_time', 0.0),
                stage_id=result.get('stage_id', 0),
                sem_ok=result.get('sem_ok', True),
                rollback_count=result.get('rollback_count', 0),
                perf=result.get('perf', {})
            )
        elif tag.compartment == 'STEP':
            # MStepReceipt for accepted steps
            return MStepReceipt(
                step_id=self.memory_contract.step_counter + 1,
                kappa=tag.kappa,
                t=result.get('t', 0.0),
                tau_step=self.clock.tau_l,
                residual_full=result.get('residuals', {}),
                enforcement_magnitude=result.get('enforcement', 0.0),
                dt_used=result.get('dt_used', 0.0),
                rollback_count=result.get('rollback_count', 0),
                gate_after=result.get('gate_result', {}),
                actions_applied=result.get('actions', [])
            )
        elif tag.compartment == 'ORCH':
            # MOrchReceipt for orchestrations
            return MOrchReceipt(
                o=result.get('o', 0),
                window_steps=result.get('window_steps', []),
                quantiles=result.get('quantiles', {}),
                dominance_histogram=result.get('dominance_histogram', {}),
                chatter_score=result.get('chatter_score', 0.0),
                regime_label=result.get('regime_label', 'unknown'),
                promotions=result.get('promotions', []),
                verification_threshold=1e-6,
                verification_norm='L2',
                min_accepted_history=result.get('min_history', 1),
                policy_hash=self.compute_policy_hash(self.policy)
            )
        else:
            # For other compartments, could generate GRStepReceipt or custom
            return None

    def compute_policy_hash(self, policy: Dict[str, Any]) -> str:
        """Compute policy hash."""
        return self.memory_contract.compute_policy_hash(policy)

    def hash_state(self, state: Any) -> str:
        """Compute hash of state for receipts."""
        state_str = json.dumps(state, sort_keys=True, default=str)
        return hashlib.md5(state_str.encode()).hexdigest()[:16]

    def validate_gate(self, eps_H: float, eps_M: float) -> bool:
        """Validate against gate thresholds."""
        ok, reason = self.gate.check_gate(eps_H, eps_M)
        if not ok:
            raise SEMFailure(f"Gate violation: {reason}")
        return ok

    def retry_logic(self, attempt_count: int) -> Tuple[bool, float]:
        """Apply retry schedules."""
        max_attempts = self.retry_schedules['max_attempts']
        dt_floor = self.retry_schedules['dt_floor']
        backoff_factor = self.retry_schedules['backoff_factor']

        if attempt_count >= max_attempts:
            raise SEMFailure(f"Max attempts ({max_attempts}) exceeded")

        # Simplified backoff
        new_dt = dt_floor * (backoff_factor ** attempt_count)
        return True, new_dt