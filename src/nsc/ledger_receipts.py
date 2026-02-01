"""Receipt generation for LEDGER validation.

This module provides the ReceiptGenerator class for generating receipts
that track program execution steps with cryptographic integrity.
"""

import uuid
import time
from typing import Dict, Any, List, Optional, Tuple
from .ledger_types import (
    Receipt, ReceiptType, LedgerSpec, GateSpec, InvariantSpec
)
from .ledger_hash import HashChain


class ReceiptGenerator:
    """Generate receipts for program execution tracking.
    
    The receipt generator creates a chain of receipts that document each
    step of program execution, with cryptographic hashing for integrity.
    
    Attributes:
        ledger_spec: The ledger specification to follow.
        run_id: Unique identifier for this run.
        receipts: List of generated receipts.
        hash_chain: HashChain instance for receipt integrity.
        step_counter: Counter for steps in this run.
    """
    
    def __init__(self, ledger_spec: LedgerSpec, run_id: Optional[str] = None):
        """Initialize the receipt generator.
        
        Args:
            ledger_spec: The LedgerSpec defining validation requirements.
            run_id: Optional run ID (generated if not provided).
        """
        self.ledger_spec = ledger_spec
        self.run_id = run_id or f"R-{uuid.uuid4().hex[:12]}"
        self.receipts: List[Receipt] = []
        self.hash_chain = HashChain(ledger_spec.hash_algorithm)
        self.step_counter = 0
    
    def _last_hash(self) -> str:
        """Get the last hash in the chain.
        
        Returns:
            The hash of the last receipt, or genesis hash if empty.
        """
        if self.receipts:
            return self.receipts[-1].hash or ""
        return self.hash_chain.genesis_hash
    
    def _finalize_receipt(self, receipt: Receipt) -> Receipt:
        """Finalize a receipt with its hash.
        
        Args:
            receipt: The receipt to finalize.
            
        Returns:
            The finalized receipt with hash set.
        """
        data = self.hash_chain.serialize_for_hash(receipt)
        receipt.hash = self.hash_chain.compute_hash(data, receipt.hash_prev)
        self.receipts.append(receipt)
        return receipt
    
    def generate_step_proposed(self,
                               intent_id: str,
                               ops: List[str],
                               residuals: Dict[str, float],
                               thread: Optional[str] = None,
                               intent_hash: Optional[str] = None) -> Receipt:
        """Generate a step proposed receipt.
        
        This receipt is generated when a computational step is proposed,
        before any validation gates are checked.
        
        Args:
            intent_id: Identifier for the intent/operation.
            ops: List of operations in this step.
            residuals: Residual values from the computation.
            thread: Optional thread identifier.
            intent_hash: Optional hash of the intent.
            
        Returns:
            The generated step proposed receipt.
        """
        self.step_counter += 1
        
        rcpt = Receipt(
            receipt_type=ReceiptType.STEP_PROPOSED,
            timestamp=time.time(),
            run_id=self.run_id,
            step_id=self.step_counter,
            thread=thread,
            intent_id=intent_id,
            intent_hash=intent_hash,
            ops=ops,
            residuals=residuals,
            status="proposed",
            hash_prev=self._last_hash(),
        )
        
        return self._finalize_receipt(rcpt)
    
    def generate_step_accepted(self,
                               proposed_receipt: Receipt,
                               gate_results: Dict[str, Any],
                               metrics: Optional[Dict[str, Any]] = None) -> Receipt:
        """Generate a step accepted receipt.
        
        This receipt is generated when a step passes all gate checks
        and is accepted for execution.
        
        Args:
            proposed_receipt: The original step proposed receipt.
            gate_results: Results from gate evaluation.
            metrics: Optional computed metrics.
            
        Returns:
            The generated step accepted receipt.
        """
        rcpt = Receipt(
            receipt_type=ReceiptType.STEP_ACCEPTED,
            timestamp=time.time(),
            run_id=self.run_id,
            step_id=proposed_receipt.step_id,
            thread=proposed_receipt.thread,
            intent_id=proposed_receipt.intent_id,
            intent_hash=proposed_receipt.intent_hash,
            ops=proposed_receipt.ops,
            gates=gate_results,
            residuals=proposed_receipt.residuals,
            metrics=metrics or {},
            status="accepted",
            hash_prev=proposed_receipt.hash,
        )
        
        return self._finalize_receipt(rcpt)
    
    def generate_step_rejected(self,
                               proposed_receipt: Receipt,
                               gate_results: Dict[str, Any],
                               reason: str = "gate_failure") -> Receipt:
        """Generate a step rejected receipt.
        
        This receipt is generated when a step fails gate checks
        and is rejected.
        
        Args:
            proposed_receipt: The original step proposed receipt.
            gate_results: Results from gate evaluation.
            reason: Reason for rejection.
            
        Returns:
            The generated step rejected receipt.
        """
        rcpt = Receipt(
            receipt_type=ReceiptType.STEP_REJECTED,
            timestamp=time.time(),
            run_id=self.run_id,
            step_id=proposed_receipt.step_id,
            thread=proposed_receipt.thread,
            intent_id=proposed_receipt.intent_id,
            ops=proposed_receipt.ops,
            gates=gate_results,
            residuals=proposed_receipt.residuals,
            status="rejected",
            metrics={"reason": reason},
            hash_prev=proposed_receipt.hash,
        )
        
        return self._finalize_receipt(rcpt)
    
    def generate_gate_pass(self,
                           gate_id: str,
                           value: float,
                           threshold: float) -> Receipt:
        """Generate a gate pass receipt.
        
        Args:
            gate_id: The gate that passed.
            value: The value that was checked.
            threshold: The threshold used.
            
        Returns:
            The generated gate pass receipt.
        """
        rcpt = Receipt(
            receipt_type=ReceiptType.GATE_PASS,
            timestamp=time.time(),
            run_id=self.run_id,
            intent_id=gate_id,
            residuals={"value": value, "threshold": threshold},
            status="pass",
            hash_prev=self._last_hash(),
        )
        
        return self._finalize_receipt(rcpt)
    
    def generate_gate_fail(self,
                           gate_id: str,
                           value: float,
                           threshold: float,
                           reason: str = "threshold_exceeded") -> Receipt:
        """Generate a gate fail receipt.
        
        Args:
            gate_id: The gate that failed.
            value: The value that was checked.
            threshold: The threshold used.
            reason: Reason for failure.
            
        Returns:
            The generated gate fail receipt.
        """
        rcpt = Receipt(
            receipt_type=ReceiptType.GATE_FAIL,
            timestamp=time.time(),
            run_id=self.run_id,
            intent_id=gate_id,
            residuals={"value": value, "threshold": threshold},
            metrics={"reason": reason},
            status="fail",
            hash_prev=self._last_hash(),
        )
        
        return self._finalize_receipt(rcpt)
    
    def generate_invariant_check(self,
                                 invariant_id: str,
                                 passed: bool,
                                 value: float,
                                 tolerance: float = 1e-10) -> Receipt:
        """Generate an invariant check receipt.
        
        Args:
            invariant_id: The invariant that was checked.
            passed: Whether the invariant passed.
            value: The value that was checked.
            tolerance: The tolerance used.
            
        Returns:
            The generated invariant check receipt.
        """
        rcpt = Receipt(
            receipt_type=ReceiptType.CHECK_INVARIANT,
            timestamp=time.time(),
            run_id=self.run_id,
            intent_id=invariant_id,
            residuals={"value": value, "tolerance": tolerance},
            status="pass" if passed else "fail",
            hash_prev=self._last_hash(),
        )
        
        return self._finalize_receipt(rcpt)
    
    def generate_checkpoint(self,
                            checkpoint_id: str,
                            state: Dict[str, Any]) -> Receipt:
        """Generate a checkpoint created receipt.
        
        Args:
            checkpoint_id: Identifier for the checkpoint.
            state: State data at checkpoint.
            
        Returns:
            The generated checkpoint receipt.
        """
        rcpt = Receipt(
            receipt_type=ReceiptType.CKPT_CREATED,
            timestamp=time.time(),
            run_id=self.run_id,
            intent_id=checkpoint_id,
            metrics={"state_keys": list(state.keys())},
            status="created",
            hash_prev=self._last_hash(),
        )
        
        return self._finalize_receipt(rcpt)
    
    def generate_rollback(self,
                          from_checkpoint: str,
                          reason: str) -> Receipt:
        """Generate a rollback executed receipt.
        
        Args:
            from_checkpoint: The checkpoint being rolled back to.
            reason: Reason for rollback.
            
        Returns:
            The generated rollback receipt.
        """
        rcpt = Receipt(
            receipt_type=ReceiptType.ROLLBACK_EXECUTED,
            timestamp=time.time(),
            run_id=self.run_id,
            intent_id=from_checkpoint,
            metrics={"reason": reason},
            status="executed",
            hash_prev=self._last_hash(),
        )
        
        return self._finalize_receipt(rcpt)
    
    def generate_run_summary(self) -> Receipt:
        """Generate a run summary receipt.
        
        This receipt summarizes the entire run and should be generated
        at the end of execution.
        
        Returns:
            The generated run summary receipt.
        """
        # Count receipts by type
        receipt_counts: Dict[str, int] = {}
        for rcpt in self.receipts:
            rcpt_type = rcpt.receipt_type.value
            receipt_counts[rcpt_type] = receipt_counts.get(rcpt_type, 0) + 1
        
        rcpt = Receipt(
            receipt_type=ReceiptType.RUN_SUMMARY,
            timestamp=time.time(),
            run_id=self.run_id,
            status="complete",
            metrics={
                "total_steps": self.step_counter,
                "total_receipts": len(self.receipts),
                "receipt_counts": receipt_counts,
                "hash_algorithm": self.ledger_spec.hash_algorithm,
            },
            hash_prev=self._last_hash(),
        )
        
        return self._finalize_receipt(rcpt)
    
    def generate_receipt_for_type(self,
                                  receipt_type: ReceiptType,
                                  **kwargs) -> Receipt:
        """Generate a receipt of a specific type.
        
        Generic method for generating receipts by type.
        
        Args:
            receipt_type: The type of receipt to generate.
            **kwargs: Additional arguments for the receipt.
            
        Returns:
            The generated receipt.
            
        Raises:
            ValueError: If receipt_type is not supported.
        """
        if receipt_type == ReceiptType.STEP_PROPOSED:
            return self.generate_step_proposed(
                kwargs.get("intent_id", ""),
                kwargs.get("ops", []),
                kwargs.get("residuals", {}),
            )
        elif receipt_type == ReceiptType.STEP_ACCEPTED:
            return self.generate_step_accepted(
                kwargs.get("proposed_receipt"),
                kwargs.get("gate_results", {}),
            )
        elif receipt_type == ReceiptType.CHECK_INVARIANT:
            return self.generate_invariant_check(
                kwargs.get("invariant_id", ""),
                kwargs.get("passed", False),
                kwargs.get("value", 0.0),
            )
        else:
            raise ValueError(f"Unsupported receipt type: {receipt_type}")
    
    def get_receipts(self) -> List[Receipt]:
        """Get all generated receipts.
        
        Returns:
            List of all receipts generated so far.
        """
        return self.receipts.copy()
    
    def get_receipts_by_type(self, receipt_type: ReceiptType) -> List[Receipt]:
        """Get receipts of a specific type.
        
        Args:
            receipt_type: The type to filter by.
            
        Returns:
            List of matching receipts.
        """
        return [r for r in self.receipts if r.receipt_type == receipt_type]
    
    def validate_chain(self) -> Tuple[bool, List[str]]:
        """Validate the hash chain of generated receipts.
        
        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        return self.hash_chain.validate_chain(self.receipts)
    
    def clear_receipts(self) -> None:
        """Clear all generated receipts and reset the chain."""
        self.receipts = []
        self.step_counter = 0
        self.hash_chain.reset()
    
    def set_run_id(self, run_id: str) -> None:
        """Set a new run ID (clears existing receipts).
        
        Args:
            run_id: The new run ID.
        """
        self.run_id = run_id
        self.clear_receipts()
