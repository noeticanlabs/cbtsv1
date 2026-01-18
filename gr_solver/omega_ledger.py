from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from typing_extensions import TypedDict
import hashlib
import json

class TimeParams(TypedDict):
    chi_alpha: float
    chi_min: float
    chi_max: float
    dt_normalization: float
    incoherence_weights: Dict[str, float]

class TimePolicy(TypedDict):
    dt_max: float
    dt_min: float
    retry_max: int
    tie_tolerance: float

class Audit(TypedDict):
    eps_UFE_rms: float
    eps_UFE_max: float
    eps_constraints: float
    invariant_drift: float
    incoherence_score: float

class Outcome(TypedDict):
    accepted: bool
    rollback_count: int
    failure_class: str

@dataclass(frozen=True)
class OmegaReceipt:
    """Per-step Ω-Ledger receipt with coherence hash chain."""
    run_id: str
    step_id: int
    t: float
    dt: float
    formulation_id: str
    gauge_id: str
    clocks: Dict[str, float]
    residuals: Dict[str, float]  # epsilon_H, M, clk, proj
    gates: Dict[str, bool]  # G1-G4
    actions: Dict[str, Any]  # dt_scale, damping_gain, etc.
    norms_metadata: Dict[str, Any]
    # LoC-PRINCIPLE-v1.0 minimal schema
    delta_norm: float  # ||delta||_Y, e.g. eps_H + eps_M
    eps_model: float  # eps^n from LoC-4
    eta_rep: float  # eta^n surrogate from LoC-6
    gamma: float  # damping rate
    clock_tau: float  # fastest safe timescale
    clock_margin: float  # dt / (c * tau)
    hash_pre: str  # hash of R(Psi^n)
    hash_post: str  # hash of R(Psi^{n+1})
    hash_step: str  # hash of step descriptor
    hash_chain: str  # hash_chain_n (same as coherence_hash)
    coherence_hash: str  # legacy, same as hash_chain
    prev_hash: str
    # LoC-Time fields
    time_params: TimeParams
    time_policy: TimePolicy
    deps: List[Tuple[str, str]]  # list of dependency edges
    audit: Audit
    outcome: Outcome

    def __post_init__(self):
        # Validation
        required_residuals = {"epsilon_H", "M", "clk", "proj"}
        if not required_residuals.issubset(self.residuals.keys()):
            raise ValueError(f"Residuals must include {required_residuals}")
        required_gates = {"G1", "G2", "G3", "G4"}
        if not required_gates.issubset(self.gates.keys()):
            raise ValueError(f"Gates must include {required_gates}")
        if self.step_id < 0:
            raise ValueError("step_id must be non-negative")
        if self.t < 0:
            raise ValueError("t must be non-negative")
        if self.dt <= 0:
            raise ValueError("dt must be positive")

class OmegaLedger:
    """Ω-Ledger managing receipts with coherence hash chain."""

    def __init__(self, run_id: str, genesis_hash: str = "genesis"):
        self.run_id = run_id
        self.receipts: List[OmegaReceipt] = []
        self.prev_hash = genesis_hash

    def emit_receipt(
        self,
        step_id: int,
        t: float,
        dt: float,
        formulation_id: str,
        gauge_id: str,
        clocks: Dict[str, float],
        residuals: Dict[str, float],
        gates: Dict[str, bool],
        actions: Dict[str, Any],
        norms_metadata: Dict[str, Any],
        delta_norm: float,
        eps_model: float,
        eta_rep: float,
        gamma: float,
        clock_tau: float,
        clock_margin: float,
        hash_pre: str,
        hash_post: str,
        hash_step: str,
        time_params: TimeParams,
        time_policy: TimePolicy,
        deps: List[Tuple[str, str]],
        audit: Audit,
        outcome: Outcome
    ) -> OmegaReceipt:
        """Emit a new Ω-receipt with coherence hash computation."""
        # Xi_n data structure for serialization
        xi_n = {
            "run_id": self.run_id,
            "step_id": step_id,
            "t": t,
            "dt": dt,
            "formulation_id": formulation_id,
            "gauge_id": gauge_id,
            "clocks": clocks,
            "residuals": residuals,
            "gates": gates,
            "actions": actions,
            "norms_metadata": norms_metadata,
            "delta_norm": delta_norm,
            "eps_model": eps_model,
            "eta_rep": eta_rep,
            "gamma": gamma,
            "clock_tau": clock_tau,
            "clock_margin": clock_margin,
            "hash_pre": hash_pre,
            "hash_post": hash_post,
            "hash_step": hash_step,
            "time_params": time_params,
            "time_policy": time_policy,
            "deps": deps,
            "audit": audit,
            "outcome": outcome
        }

        # Serialize Xi_n deterministically
        serialized_xi = json.dumps(xi_n, sort_keys=True, separators=(',', ':'))

        # Coherence hash: HASH(serialize(Xi_n) || prev_hash_{n-1})
        coherence_hash = hashlib.sha256(
            (serialized_xi + self.prev_hash).encode('utf-8')
        ).hexdigest()

        # Create receipt
        receipt = OmegaReceipt(
            run_id=self.run_id,
            step_id=step_id,
            t=t,
            dt=dt,
            formulation_id=formulation_id,
            gauge_id=gauge_id,
            clocks=clocks,
            residuals=residuals,
            gates=gates,
            actions=actions,
            norms_metadata=norms_metadata,
            delta_norm=delta_norm,
            eps_model=eps_model,
            eta_rep=eta_rep,
            gamma=gamma,
            clock_tau=clock_tau,
            clock_margin=clock_margin,
            hash_pre=hash_pre,
            hash_post=hash_post,
            hash_step=hash_step,
            hash_chain=coherence_hash,
            coherence_hash=coherence_hash,
            prev_hash=self.prev_hash,
            time_params=time_params,
            time_policy=time_policy,
            deps=deps,
            audit=audit,
            outcome=outcome
        )

        # Append and update chain
        self.receipts.append(receipt)
        self.prev_hash = coherence_hash

        return receipt

    def verify_chain(self) -> bool:
        """Verify the coherence hash chain integrity."""
        prev_hash = self.receipts[0].prev_hash if self.receipts else "genesis"
        for receipt in self.receipts:
            # Recompute hash
            xi_n = {
                "run_id": receipt.run_id,
                "step_id": receipt.step_id,
                "t": receipt.t,
                "dt": receipt.dt,
                "formulation_id": receipt.formulation_id,
                "gauge_id": receipt.gauge_id,
                "clocks": receipt.clocks,
                "residuals": receipt.residuals,
                "gates": receipt.gates,
                "actions": receipt.actions,
                "norms_metadata": receipt.norms_metadata,
                "delta_norm": receipt.delta_norm,
                "eps_model": receipt.eps_model,
                "eta_rep": receipt.eta_rep,
                "gamma": receipt.gamma,
                "clock_tau": receipt.clock_tau,
                "clock_margin": receipt.clock_margin,
                "hash_pre": receipt.hash_pre,
                "hash_post": receipt.hash_post,
                "hash_step": receipt.hash_step,
                "time_params": receipt.time_params,
                "time_policy": receipt.time_policy,
                "deps": receipt.deps,
                "audit": receipt.audit,
                "outcome": receipt.outcome
            }
            serialized_xi = json.dumps(xi_n, sort_keys=True, separators=(',', ':'))
            computed_hash = hashlib.sha256(
                (serialized_xi + prev_hash).encode('utf-8')
            ).hexdigest()
            if computed_hash != receipt.coherence_hash:
                return False
            prev_hash = receipt.coherence_hash
        return True

    def get_receipts(self) -> List[OmegaReceipt]:
        """Get all receipts."""
        return self.receipts.copy()