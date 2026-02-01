# NSC Models - Ledger Model
# Invariants, gates, receipts for audit and coherence

"""
NSC_Ledger - Ledger/Audit Model

This module provides ledger operations for:
- Gate checking (SEM, CONS, PHY barriers)
- Receipt generation (hash-chained audit trail)
- Invariant enforcement (constraint preservation)
- Proof obligation tracking

Supported Models:
- LEDGER: Gate checking, receipt generation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import hashlib
import json


# =============================================================================
# Gate Types and Classes
# =============================================================================

class GateType(Enum):
    """Classification of coherence gates."""
    SEM = "sem"      # Semantic (hard barriers)
    CONS = "cons"    # Constraint preservation
    PHY = "phy"      # Physical validity


class GateResult(Enum):
    """Result of gate evaluation."""
    PASS = "pass"
    SOFT_FAIL = "soft_fail"  # Warning, accept
    HARD_FAIL = "hard_fail"  # Rollback required


@dataclass
class GateThreshold:
    """Threshold configuration for a gate."""
    name: str
    gate_type: GateType
    hard_max: float      # Hard failure if exceeded
    soft_max: float      # Soft failure if exceeded
    unit: str = "1"      # Dimensionless by default


@dataclass
class GateEvaluation:
    """Result of gate evaluation."""
    gate_name: str
    gate_type: GateType
    value: float
    result: GateResult
    margin: float        # Distance to threshold
    reason: Optional[str] = None


@dataclass
class GatePolicy:
    """Collection of gate thresholds for a simulation."""
    policy_name: str
    thresholds: Dict[str, GateThreshold] = field(default_factory=dict)
    
    def add_threshold(self, threshold: GateThreshold):
        """Add a gate threshold."""
        self.thresholds[threshold.name] = threshold
    
    def evaluate(self, name: str, value: float) -> GateEvaluation:
        """Evaluate a value against the gate policy."""
        threshold = self.thresholds.get(name)
        if threshold is None:
            return GateEvaluation(
                gate_name=name,
                gate_type=GateType.PHY,
                value=value,
                result=GateResult.PASS,
                margin=float('inf'),
                reason="No threshold defined"
            )
        
        if value > threshold.hard_max:
            margin = threshold.hard_max - value
            return GateEvaluation(
                gate_name=name,
                gate_type=threshold.gate_type,
                value=value,
                result=GateResult.HARD_FAIL,
                margin=margin,
                reason=f"Value {value} exceeds hard max {threshold.hard_max}"
            )
        elif value > threshold.soft_max:
            margin = threshold.hard_max - value
            return GateEvaluation(
                gate_name=name,
                gate_type=threshold.gate_type,
                value=value,
                result=GateResult.SOFT_FAIL,
                margin=margin,
                reason=f"Value {value} exceeds soft max {threshold.soft_max}"
            )
        else:
            margin = threshold.hard_max - value
            return GateEvaluation(
                gate_name=name,
                gate_type=threshold.gate_type,
                value=value,
                result=GateResult.PASS,
                margin=margin,
                reason=None
            )


# =============================================================================
# Receipt System
# =============================================================================

@dataclass
class Receipt:
    """Hash-chained receipt for simulation audit.
    
    Each receipt commits to the previous receipt via hash chain,
    ensuring immutable audit trail.
    """
    receipt_id: str
    step_id: int
    timestamp: float
    event_type: str
    input_hash: str
    output_hash: str
    prev_receipt_hash: str
    data: Dict[str, Any]
    
    @classmethod
    def create(cls, step_id: int, event_type: str, data: Dict[str, Any],
               prev_receipt_hash: str = "") -> 'Receipt':
        """Create a new receipt."""
        import time
        receipt_id = hashlib.sha256(
            f"{step_id}{event_type}{time.time()}".encode()
        ).hexdigest()[:16]
        
        input_hash = hashlib.sha256(json.dumps(data).encode()).hexdigest()
        output_hash = ""
        
        return cls(
            receipt_id=receipt_id,
            step_id=step_id,
            timestamp=time.time(),
            event_type=event_type,
            input_hash=input_hash,
            output_hash=output_hash,
            prev_receipt_hash=prev_receipt_hash,
            data=data
        )
    
    def seal(self, output_hash: str):
        """Seal the receipt with output hash."""
        self.output_hash = output_hash
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'receipt_id': self.receipt_id,
            'step_id': self.step_id,
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'input_hash': self.input_hash,
            'output_hash': self.output_hash,
            'prev_receipt_hash': self.prev_receipt_hash,
            'data': self.data
        }


@dataclass
class ReceiptChain:
    """Chain of receipts for audit trail."""
    receipts: List[Receipt] = field(default_factory=list)
    
    def add(self, receipt: Receipt):
        """Add receipt to chain."""
        if self.receipts:
            receipt.prev_receipt_hash = self.receipts[-1].receipt_id
        self.receipts.append(receipt)
    
    def verify(self) -> Tuple[bool, List[str]]:
        """Verify chain integrity."""
        errors = []
        for i, receipt in enumerate(self.receipts):
            if i > 0:
                prev = self.receipts[i-1]
                if receipt.prev_receipt_hash != prev.receipt_id:
                    errors.append(f"Chain broken at receipt {i}")
        return len(errors) == 0, errors
    
    def get_last_hash(self) -> str:
        """Get hash of last receipt."""
        if self.receipts:
            return self.receipts[-1].receipt_id
        return ""


# =============================================================================
# Invariant System
# =============================================================================

@dataclass
class InvariantSpec:
    """Specification of an invariant to enforce."""
    invariant_id: str
    description: str
    gate_type: GateType
    receipt_field: str
    computation: str  # How to compute the invariant value


INVARIANT_REGISTRY = {
    'hamiltonian_constraint': {
        'id': 'N:INV.gr.hamiltonian_constraint',
        'description': 'Hamiltonian constraint residual zero',
        'gate_type': 'HARD',
        'receipt_field': 'residuals.eps_H',
        'computation': 'compute_hamiltonian'
    },
    'momentum_constraint': {
        'id': 'N:INV.gr.momentum_constraint',
        'description': 'Momentum constraint residual zero',
        'gate_type': 'SOFT',
        'receipt_field': 'residuals.eps_M',
        'computation': 'compute_momentum'
    },
    'div_free': {
        'id': 'N:INV.ns.div_free',
        'description': 'Velocity divergence-free',
        'gate_type': 'HARD',
        'receipt_field': 'residuals.eps_div',
        'computation': 'compute_divergence'
    },
    'energy_nonincreasing': {
        'id': 'N:INV.ns.energy_nonincreasing',
        'description': 'Energy bounded by forcing/dissipation',
        'gate_type': 'SOFT',
        'receipt_field': 'residuals.eps_energy',
        'computation': 'compute_energy_balance'
    },
    'stage_coherence': {
        'id': 'N:INV.clock.stage_coherence',
        'description': 'Stage times consistent',
        'gate_type': 'HARD',
        'receipt_field': 'metrics.delta_stage_t',
        'computation': 'check_stage_times'
    },
    'hash_chain_intact': {
        'id': 'N:INV.ledger.hash_chain_intact',
        'description': 'Receipt chain unbroken',
        'gate_type': 'HARD',
        'receipt_field': 'chain.verified',
        'computation': 'verify_chain'
    }
}


# =============================================================================
# NSC_Ledger Dialect
# =============================================================================

class NSC_Ledger_Dialect:
    """NSC_Ledger Dialect for audit and invariant enforcement.
    
    Provides:
    - Gate checking and evaluation
    - Receipt generation and chain management
    - Invariant registry and enforcement
    """
    
    name = "NSC_models.ledger"
    version = "1.0"
    
    mandatory_models = ['LEDGER']
    
    gate_types = GateType
    gate_results = GateResult
    
    def __init__(self):
        """Initialize ledger dialect."""
        self.receipt_chain = ReceiptChain()
        self.default_policy = self._create_default_policy()
    
    def _create_default_policy(self) -> GatePolicy:
        """Create default gate policy."""
        policy = GatePolicy("default")
        
        # GR constraints
        policy.add_threshold(GateThreshold(
            "eps_H", GateType.CONS, 1e-6, 1e-8
        ))
        policy.add_threshold(GateThreshold(
            "eps_M", GateType.CONS, 1e-5, 1e-7
        ))
        
        # NS constraints
        policy.add_threshold(GateThreshold(
            "eps_div", GateType.CONS, 1e-8, 1e-10
        ))
        policy.add_threshold(GateThreshold(
            "eps_energy", GateType.SEM, 1e-6, 1e-8
        ))
        
        # Physical validity
        policy.add_threshold(GateThreshold(
            "eps_clk", GateType.PHY, 1e-8, 1e-10
        ))
        
        return policy
    
    def create_gate_policy(self, name: str, **kwargs) -> GatePolicy:
        """Create a gate policy with custom thresholds."""
        policy = GatePolicy(name)
        for gate_name, value in kwargs.items():
            if isinstance(value, (int, float)):
                policy.add_threshold(GateThreshold(
                    gate_name, GateType.PHY, value, value * 0.1
                ))
        return policy
    
    def evaluate_gates(self, metrics: Dict[str, float], 
                       policy: Optional[GatePolicy] = None) -> List[GateEvaluation]:
        """Evaluate all metrics against gate policy."""
        if policy is None:
            policy = self.default_policy
        
        evaluations = []
        for metric_name, value in metrics.items():
            evaluation = policy.evaluate(metric_name, value)
            evaluations.append(evaluation)
        
        return evaluations
    
    def create_receipt(self, step_id: int, event_type: str, 
                       data: Dict[str, Any]) -> Receipt:
        """Create a new receipt."""
        prev_hash = self.receipt_chain.get_last_hash()
        receipt = Receipt.create(step_id, event_type, data, prev_hash)
        self.receipt_chain.add(receipt)
        return receipt
    
    def verify_chain(self) -> Tuple[bool, List[str]]:
        """Verify receipt chain integrity."""
        return self.receipt_chain.verify()
    
    def get_invariant(self, name: str) -> Optional[InvariantSpec]:
        """Get invariant specification."""
        if name in INVARIANT_REGISTRY:
            spec = INVARIANT_REGISTRY[name]
            return InvariantSpec(
                invariant_id=spec['id'],
                description=spec['description'],
                gate_type=GateType.SEM if spec['gate_type'] == 'HARD' else GateType.CONS,
                receipt_field=spec['receipt_field'],
                computation=spec['computation']
            )
        return None
    
    def list_invariants(self) -> List[str]:
        """List registered invariants."""
        return list(INVARIANT_REGISTRY.keys())


# Export singleton
NSC_ledger = NSC_Ledger_Dialect()
