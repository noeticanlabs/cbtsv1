"""LEDGER types for NSC-M3L receipt validation and invariant checking.

This module defines the core types for the LEDGER model, including:
- Receipt types for tracking program execution
- Gate specifications for threshold-based validation
- Invariant specifications for mathematical constraints
- Ledger specification for complete program validation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ReceiptType(Enum):
    """Enumeration of all receipt types per AEONICA specification."""
    STEP_PROPOSED = "A:RCPT.step.proposed"
    STEP_ACCEPTED = "A:RCPT.step.accepted"
    STEP_REJECTED = "A:RCPT.step.rejected"
    GATE_PASS = "A:RCPT.gate.pass"
    GATE_FAIL = "A:RCPT.gate.fail"
    CHECK_INVARIANT = "A:RCPT.check.invariant"
    CKPT_CREATED = "A:RCPT.ckpt.created"
    ROLLBACK_EXECUTED = "A:RCPT.rollback.executed"
    RUN_SUMMARY = "A:RCPT.run.summary"


@dataclass
class Receipt:
    """Base receipt class per GML spec for tracking execution events.
    
    Attributes:
        receipt_type: The type of receipt (step proposed, accepted, etc.)
        timestamp: Unix timestamp of receipt generation
        run_id: Unique identifier for the run
        step_id: Optional step number within the run
        thread: Optional thread identifier for concurrent execution
        intent_id: Optional intent/operation identifier
        intent_hash: Hash of the intent for integrity
        ops: List of operations in this step
        gates: Gate evaluation results
        residuals: Residual values from computations
        metrics: Performance/computation metrics
        status: Status of the step (proposed, accepted, rejected)
        hash_prev: Hash of the previous receipt in chain
        hash: Hash of this receipt
    """
    receipt_type: ReceiptType
    timestamp: float
    run_id: str
    step_id: Optional[int] = None
    thread: Optional[str] = None
    intent_id: Optional[str] = None
    intent_hash: Optional[str] = None
    ops: List[str] = field(default_factory=list)
    gates: Dict[str, Any] = field(default_factory=dict)
    residuals: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = "proposed"
    hash_prev: Optional[str] = None
    hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary for serialization."""
        return {
            "receipt_type": self.receipt_type.value,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "thread": self.thread,
            "intent_id": self.intent_id,
            "intent_hash": self.intent_hash,
            "ops": self.ops,
            "gates": self.gates,
            "residuals": self.residuals,
            "metrics": self.metrics,
            "status": self.status,
            "hash_prev": self.hash_prev,
            "hash": self.hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Receipt":
        """Create receipt from dictionary."""
        data = data.copy()
        data["receipt_type"] = ReceiptType(data["receipt_type"])
        return cls(**data)


@dataclass
class GateSpec:
    """Specification for a gate validation threshold.
    
    Gates define acceptance conditions for computational steps,
    including thresholds, hysteresis bands, and comparison operations.
    
    Attributes:
        gate_id: Unique identifier for this gate
        threshold: The threshold value for comparison
        hysteresis: Optional hysteresis band width
        comparison: Comparison operator (le, ge, lt, gt, eq)
        window: Optional window size for rolling average checks
    """
    gate_id: str
    threshold: float
    hysteresis: float = 0.0
    comparison: str = "le"  # le, ge, lt, gt, eq
    window: Optional[int] = None  # For rolling average
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert gate spec to dictionary."""
        return {
            "gate_id": self.gate_id,
            "threshold": self.threshold,
            "hysteresis": self.hysteresis,
            "comparison": self.comparison,
            "window": self.window,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GateSpec":
        """Create gate spec from dictionary."""
        return cls(**data)


@dataclass
class InvariantSpec:
    """Specification for an invariant constraint.
    
    Invariants define mathematical constraints that must hold
    throughout computation, with configurable tolerance bands.
    
    Attributes:
        invariant_id: Unique identifier for this invariant
        tolerance_abs: Absolute tolerance for violation detection
        tolerance_rel: Relative tolerance for violation detection
        gate_key: Key for associated gate if any
        source_metric: Name of the source metric to check
    """
    invariant_id: str
    tolerance_abs: float = 1e-10
    tolerance_rel: float = 1e-8
    gate_key: str = ""
    source_metric: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert invariant spec to dictionary."""
        return {
            "invariant_id": self.invariant_id,
            "tolerance_abs": self.tolerance_abs,
            "tolerance_rel": self.tolerance_rel,
            "gate_key": self.gate_key,
            "source_metric": self.source_metric,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InvariantSpec":
        """Create invariant spec from dictionary."""
        return cls(**data)


@dataclass
class LedgerSpec:
    """Complete ledger specification for a program.
    
    The ledger spec defines all validation requirements for a program,
    including required invariants, gates, receipts, and proof obligations.
    
    Attributes:
        invariants: List of invariant specifications
        gates: List of gate specifications
        required_receipts: List of receipt types that must be generated
        proof_obligations: List of proof obligation identifiers
        hash_algorithm: Hash algorithm for receipt chain (default: sha256)
    """
    invariants: List[InvariantSpec] = field(default_factory=list)
    gates: List[GateSpec] = field(default_factory=list)
    required_receipts: List[ReceiptType] = field(default_factory=list)
    proof_obligations: List[str] = field(default_factory=list)
    hash_algorithm: str = "sha256"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ledger spec to dictionary."""
        return {
            "invariants": [inv.to_dict() for inv in self.invariants],
            "gates": [gate.to_dict() for gate in self.gates],
            "required_receipts": [r.value for r in self.required_receipts],
            "proof_obligations": self.proof_obligations,
            "hash_algorithm": self.hash_algorithm,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LedgerSpec":
        """Create ledger spec from dictionary."""
        return cls(
            invariants=[InvariantSpec.from_dict(i) for i in data.get("invariants", [])],
            gates=[GateSpec.from_dict(g) for g in data.get("gates", [])],
            required_receipts=[ReceiptType(r) for r in data.get("required_receipts", [])],
            proof_obligations=data.get("proof_obligations", []),
            hash_algorithm=data.get("hash_algorithm", "sha256"),
        )
    
    def get_gate(self, gate_id: str) -> Optional[GateSpec]:
        """Get a gate specification by ID."""
        for gate in self.gates:
            if gate.gate_id == gate_id:
                return gate
        return None
    
    def get_invariant(self, invariant_id: str) -> Optional[InvariantSpec]:
        """Get an invariant specification by ID."""
        for inv in self.invariants:
            if inv.invariant_id == invariant_id:
                return inv
        return None


@dataclass
class GateResult:
    """Result of gate evaluation.
    
    Attributes:
        gate_id: The gate that was evaluated
        passed: Whether the gate passed
        value: The value that was checked
        threshold: The threshold used for comparison
        comparison: The comparison operator used
        status: Status string (pass, fail, review)
        hysteresis: Hysteresis band if applicable
        window_average: Rolling average if window specified
    """
    gate_id: str
    passed: bool
    value: float
    threshold: float
    comparison: str
    status: str = "pass"
    hysteresis: Optional[float] = None
    window_average: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "gate_id": self.gate_id,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "status": self.status,
        }
        if self.hysteresis is not None:
            result["hysteresis"] = self.hysteresis
        if self.window_average is not None:
            result["window_average"] = self.window_average
        return result


@dataclass
class InvariantResult:
    """Result of invariant checking.
    
    Attributes:
        invariant_id: The invariant that was checked
        passed: Whether the invariant passed
        value: The value that was checked
        tolerance_abs: Absolute tolerance used
        tolerance_rel: Relative tolerance used
        status: Status string
        reason: Explanation of the result
    """
    invariant_id: str
    passed: bool
    value: float
    tolerance_abs: float
    tolerance_rel: float
    status: str = "pass"
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "invariant_id": self.invariant_id,
            "passed": self.passed,
            "value": self.value,
            "tolerance_abs": self.tolerance_abs,
            "tolerance_rel": self.tolerance_rel,
            "status": self.status,
            "reason": self.reason,
        }
