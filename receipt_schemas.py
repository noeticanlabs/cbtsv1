from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np

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

    def __post_init__(self):
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