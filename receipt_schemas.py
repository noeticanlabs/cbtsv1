from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np
import hashlib
import json

class SEMFailure(Exception):
    """SEM hard failure - immediate abort."""
    pass

@dataclass(frozen=True)
class Kappa:
    """Typed kappa for unambiguous time semantics."""
    o: int      # orchestrator window index
    s: int      # accepted step index
    mu: Optional[int]  # micro stage / inner iter

    def __post_init__(self):
        if self.o < 0:
            raise SEMFailure(f"Kappa.o must be non-negative, got {self.o}")
        if self.s < 0:
            raise SEMFailure(f"Kappa.s must be non-negative, got {self.s}")
        if self.mu is not None and self.mu < 0:
            raise SEMFailure(f"Kappa.mu must be non-negative if present, got {self.mu}")

@dataclass
class MSolveReceipt:
    """Attempt receipts (every attempt, accepted or not)."""
    attempt_id: int
    kappa: Kappa
    t: float
    tau_attempt: int
    residual_proxy: Dict[str, Dict[str, float]]  # {domain: {scale: normalized_value}}
    dt_cap_min: float
    dominant_thread: tuple
    actions: List[Dict[str, Any]]
    policy_hash: str
    state_hash: str  # Cheap fingerprint
    stage_time: float
    stage_id: int
    sem_ok: bool
    rollback_count: int
    perf: Dict[str, Any]
    
    # Forensic fields for audit-grade recovery
    gate_kind: str = "state"  # constraint, state, rate, nonfinite, uninitialized
    gate_reason: str = ""  # Specific failure reason
    hard_fail: bool = False  # Whether this was a hard fail
    retry_index: int = 0  # 0 for first attempt
    dt_before: float = 0.0  # dt before this attempt
    dt_after: float = 0.0  # dt after this attempt (may be reduced)
    snapshot_id: str = ""  # Hash of (state, clock) snapshot
    regime_hash: str = ""  # Current regime hash
    repair_actions_allowed: List[str] = None  # List of allowed repair action names
    repair_action_chosen: Optional[str] = None  # Chosen repair action
    
    def __post_init__(self):
        if self.repair_actions_allowed is None:
            self.repair_actions_allowed = []
        if self.attempt_id < 0:
            raise SEMFailure(f"attempt_id must be non-negative, got {self.attempt_id}")
        if not np.isfinite(self.t):
            raise SEMFailure(f"t must be finite, got {self.t}")
        if self.tau_attempt < 0:
            raise SEMFailure(f"tau_attempt must be non-negative, got {self.tau_attempt}")
        if self.dt_cap_min <= 0 or not np.isfinite(self.dt_cap_min):
            raise SEMFailure(f"dt_cap_min must be positive finite, got {self.dt_cap_min}")
        if self.stage_time < 0 or not np.isfinite(self.stage_time):
            raise SEMFailure(f"stage_time must be non-negative finite, got {self.stage_time}")
        if self.stage_id < 0:
            raise SEMFailure(f"stage_id must be non-negative, got {self.stage_id}")
        if self.rollback_count < 0:
            raise SEMFailure(f"rollback_count must be non-negative, got {self.rollback_count}")
        if not self.policy_hash:
            raise SEMFailure("policy_hash cannot be empty")
        if not self.state_hash:
            raise SEMFailure("state_hash cannot be empty")
        # Validate gate_kind
        valid_kinds = {"constraint", "state", "rate", "nonfinite", "uninitialized"}
        if self.gate_kind not in valid_kinds:
            raise SEMFailure(f"gate_kind must be one of {valid_kinds}, got {self.gate_kind}")
        # Check residual_proxy
        for domain, scales in self.residual_proxy.items():
            if domain is None:
                raise SEMFailure("residual_proxy domain cannot be None")
            for scale, val in scales.items():
                if scale is None or not np.isfinite(val):
                    raise SEMFailure(f"residual_proxy {domain}.{scale} must be finite, got {val}")

@dataclass
class MStepReceipt:
    """Only accepted steps."""
    step_id: int
    kappa: Kappa
    t: float
    tau_step: int
    residual_full: Dict[str, Dict[str, float]]
    enforcement_magnitude: float
    dt_used: float
    rollback_count: int
    gate_after: Dict[str, Any]
    actions_applied: List[Dict[str, Any]]

    def __post_init__(self):
        if self.step_id < 0:
            raise SEMFailure(f"step_id must be non-negative, got {self.step_id}")
        if not np.isfinite(self.t):
            raise SEMFailure(f"t must be finite, got {self.t}")
        if self.tau_step < 0:
            raise SEMFailure(f"tau_step must be non-negative, got {self.tau_step}")
        if self.enforcement_magnitude < 0 or not np.isfinite(self.enforcement_magnitude):
            raise SEMFailure(f"enforcement_magnitude must be non-negative finite, got {self.enforcement_magnitude}")
        if self.dt_used <= 0 or not np.isfinite(self.dt_used):
            raise SEMFailure(f"dt_used must be positive finite, got {self.dt_used}")
        if self.rollback_count < 0:
            raise SEMFailure(f"rollback_count must be non-negative, got {self.rollback_count}")
        # Check residual_full
        for domain, scales in self.residual_full.items():
            if domain is None:
                raise SEMFailure("residual_full domain cannot be None")
            for scale, val in scales.items():
                if scale is None or not np.isfinite(val):
                    raise SEMFailure(f"residual_full {domain}.{scale} must be finite, got {val}")

@dataclass
class MOrchReceipt:
    """Canon promotions."""
    o: int
    window_steps: List[int]
    quantiles: Dict[str, Dict[str, float]]
    dominance_histogram: Dict[str, int]
    chatter_score: float
    regime_label: str
    promotions: List[Dict[str, Any]]
    verification_threshold: float
    verification_norm: str
    min_accepted_history: int
    policy_hash: str

    def __post_init__(self):
        if self.o < 0:
            raise SEMFailure(f"o must be non-negative, got {self.o}")
        for ws in self.window_steps:
            if ws < 0:
                raise SEMFailure(f"window_steps must be non-negative, got {ws}")
        for dom, hist in self.dominance_histogram.items():
            if dom is None or hist < 0:
                raise SEMFailure(f"dominance_histogram {dom} must be non-negative, got {hist}")
        if self.chatter_score < 0 or not np.isfinite(self.chatter_score):
            raise SEMFailure(f"chatter_score must be non-negative finite, got {self.chatter_score}")
        if not self.regime_label:
            raise SEMFailure("regime_label cannot be empty")
        if self.verification_threshold <= 0 or not np.isfinite(self.verification_threshold):
            raise SEMFailure(f"verification_threshold must be positive finite, got {self.verification_threshold}")
        if not self.verification_norm:
            raise SEMFailure("verification_norm cannot be empty")
        if self.min_accepted_history < 0:
            raise SEMFailure(f"min_accepted_history must be non-negative, got {self.min_accepted_history}")
        if not self.policy_hash:
            raise SEMFailure("policy_hash cannot be empty")
        # Check quantiles
        for domain, scales in self.quantiles.items():
            if domain is None:
                raise SEMFailure("quantiles domain cannot be None")
            for scale, val in scales.items():
                if scale is None or not np.isfinite(val):
                    raise SEMFailure(f"quantiles {domain}.{scale} must be finite, got {val}")

@dataclass
class GRStepReceipt:
    """Receipt for GR/NR solver steps."""
    module_id: str
    dep_closure_hash: str
    compiler: str
    target: str  # "loc-gr-nr"
    step_id: int
    tau_n: int
    dt: float
    stage_count: int
    retry_count: int
    thread_id: str  # e.g., "PHY.step.act"
    eps_H: float
    eps_M: float
    R: float
    H: float
    dH: float
    state_hash_before: str
    state_hash_after: str
    policy_hash: str
    prev: Optional[str]  # hash chain
    id: str  # hash of this receipt

    def __post_init__(self):
        if not self.module_id:
            raise ValueError("module_id cannot be empty")
        if not self.dep_closure_hash:
            raise ValueError("dep_closure_hash cannot be empty")
        if not self.compiler:
            raise ValueError("compiler cannot be empty")
        if self.target != "loc-gr-nr":
            raise ValueError(f"target must be 'loc-gr-nr', got {self.target}")
        if self.step_id < 0:
            raise ValueError(f"step_id must be non-negative, got {self.step_id}")
        if self.tau_n < 0:
            raise ValueError(f"tau_n must be non-negative, got {self.tau_n}")
        if self.dt <= 0 or not np.isfinite(self.dt):
            raise ValueError(f"dt must be positive finite, got {self.dt}")
        if self.stage_count < 0:
            raise ValueError(f"stage_count must be non-negative, got {self.stage_count}")
        if self.retry_count < 0:
            raise ValueError(f"retry_count must be non-negative, got {self.retry_count}")
        if not self.thread_id:
            raise ValueError("thread_id cannot be empty")
        # Physics values must be finite
        for name, val in [("eps_H", self.eps_H), ("eps_M", self.eps_M), ("R", self.R), ("H", self.H), ("dH", self.dH)]:
            if not np.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")
        if not self.state_hash_before:
            raise ValueError("state_hash_before cannot be empty")
        if not self.state_hash_after:
            raise ValueError("state_hash_after cannot be empty")
        if not self.policy_hash:
            raise ValueError("policy_hash cannot be empty")
        if not self.id:
            raise ValueError("id cannot be empty")

    @classmethod
    def create(cls, **kwargs):
        # Compute id if not provided
        if 'id' not in kwargs:
            receipt_data = {k: v for k, v in kwargs.items() if k != 'id'}
            canonical = json.dumps(receipt_data, sort_keys=True, separators=(',', ':'))
            kwargs['id'] = hashlib.sha256(canonical.encode()).hexdigest()
        return cls(**kwargs)


@dataclass
class OmegaReceipt:
    """Unified receipt with hash chaining."""
    prev: Optional[str]  # Previous receipt id, null for genesis
    tier: str  # Receipt tier: "msolve", "mstep", "morch", "grstep"
    record: Dict[str, Any]  # Specific receipt data
    id: str  # SHA256 hash of {prev, tier, record}

    def __post_init__(self):
        if not self.tier:
            raise SEMFailure("tier cannot be empty")
        if self.id != self.compute_id(self.prev, self.tier, self.record):
            raise SEMFailure("id does not match computed hash")
        # Note: prev can be None for genesis

    @staticmethod
    def compute_id(prev: Optional[str], tier: str, record: Dict[str, Any]) -> str:
        """Compute receipt id as sha256(canonical_json({prev, tier, record})).hex"""
        data = {
            "prev": prev,
            "tier": tier,
            "record": record
        }
        canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode()).hexdigest()

    @classmethod
    def create(cls, prev: Optional[str], tier: str, record: Dict[str, Any]) -> 'OmegaReceipt':
        """Create OmegaReceipt with computed id."""
        id_hash = cls.compute_id(prev, tier, record)
        return cls(prev=prev, tier=tier, record=record, id=id_hash)


def validate_receipt_chain(receipts: List[OmegaReceipt]) -> bool:
    """Validate receipt chain by checking hashes and prev links."""
    if not receipts:
        return True
    # Check first receipt: prev should be None or some genesis
    # But since prev can be None, and for now, assume the first has prev=None
    prev_id = None
    for receipt in receipts:
        if receipt.prev != prev_id:
            return False
        # Verify id matches
        computed_id = OmegaReceipt.compute_id(receipt.prev, receipt.tier, receipt.record)
        if receipt.id != computed_id:
            return False
        prev_id = receipt.id
    return True
