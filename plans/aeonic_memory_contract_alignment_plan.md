# Aeonic Memory Contract v1.0 Alignment Plan

## Executive Summary

Align the existing Aeonic Memory implementation (`aeonic_memory_bank.py`, `aeonic_receipts.py`) with the new Aeonic Memory Contract v1.0 specification. The current tier-based system (tiers 1,2,3) will be maintained for backward compatibility, but new contract-compliant interfaces will be added to map to the required taxonomy (M_solve, M_step, M_orch).

## Current Implementation Gaps

### Memory Taxonomy
- **Current**: Tiered storage (1,2,3) with generic Records
- **Spec**: Dedicated memories: M_solve (ring buffer), M_step (accepted history), M_orch (canon)
- **Gap**: No semantic separation by epistemic role

### PhaseLoom Integration
- **Current**: Partial implementation in `phaseloom_threads_gr.py` (7 threads, not 27)
- **Spec**: 27-thread lattice (3 domains × 3 scales × 3 responses)
- **Gap**: Missing 27-thread structure, dt arbitration, dominant thread detection

### Receipt Schemas
- **Current**: Basic JSONL logging with minimal fields
- **Spec**: Detailed schemas per memory type with stamps, residuals, actions, policy hashes
- **Gap**: Insufficient detail for coherence verification

### SEM Safety
- **Current**: No SEM enforcement
- **Spec**: SEM violations as hard barriers
- **Gap**: Missing policy fingerprinting, rejection logic

### Promotion & Pruning
- **Current**: TTL-based expiration + V-score eviction
- **Spec**: Promotion contract, canon append-only, regime-based invalidation
- **Gap**: No earned truth promotion, no epistemic separation

## Detailed Design Plan

### 1. Memory Taxonomy Mapping

Keep existing `AeonicMemoryBank` tiers but add contract interface with strict attempt/accepted separation:

```python
@dataclass(frozen=True)
class Kappa:
    o: int      # orchestrator window index
    s: int      # accepted step index
    mu: int | None  # micro stage / inner iter

class AeonicMemoryContract:
    def __init__(self, memory_bank: AeonicMemoryBank, phaseloom: PhaseLoom27):
        # Map tiers to memories
        self.M_solve = memory_bank.tiers[1]  # Ring buffer for attempts
        self.M_step = memory_bank.tiers[2]   # Only accepted steps
        self.M_orch = memory_bank.tiers[3]   # Canon promotions

    # Contract-compliant methods
    def put_attempt_receipt(self, kappa: Kappa, receipt_data): ...  # attempt_id increments always
    def put_step_receipt(self, kappa: Kappa, receipt_data): ...     # step_id increments only on acceptance
    def put_orch_receipt(self, o, receipt_data): ...
    def get_by_kappa(self, kappa_range): ...
    def promote_to_canon(self, step_receipts, gate_result): ...

    # SEM Hard Rules
    def validate_min_accepted_history(self, min_steps):  # Must check len(M_step) >= min_steps before verification
        return len(self.M_step) >= min_steps

    def abort_on_state_gate_no_repair(self, gate):  # Fast-exit if state gate with no repair action
        if gate['kind'] == 'state' and not any(action['repair'] for action in gate.get('actions_allowed', [])):
            raise SEMFailure("State gate violation with no repair action - abort immediately")
```

### 2. PhaseLoom 27-Thread Lattice

Extend `phaseloom_threads_gr.py` to full 27 threads:

```python
class PhaseLoom27:
    DOMAINS = ['PHY', 'CONS', 'SEM']
    SCALES = ['L', 'M', 'H']
    RESPONSES = ['R0', 'R1', 'R2']

    def __init__(self):
        self.threads = {}  # (domain, scale, response) -> ThreadState

    def compute_residuals(self, state, geometry, gauge):
        # Full vs proxy residuals per domain/scale
        # SEM: check prereqs initialized, no NaN/Inf
        pass

    def arbitrate_dt(self, residuals, stiffness):
        # Min over dt_caps, find dominant thread
        pass

    def get_gate_classification(self, dominant_thread, residual):
        # Return structured gate: {'kind': 'dt'|'state'|'sem', 'code': str, 'action_allowed': list}
        # dt: CFL, etc. -> shrink_dt
        # state: H_tail, det_tgamma -> repair_state or abort
        # sem: policy_hash_change -> abort
        pass

    def get_rails(self, dominant_thread):
        # Return corrective actions based on gate kind
        pass
```

Thread indices: T[D,S,C] for D∈{PHY,CONS,SEM}, S∈{L,M,H}, C∈{R0,R1,R2}

### 3. Receipt Schema Classes

Detailed dataclasses for each receipt type with identity keys for caching. Residual proxy defined as:

\[
r_{D,S}^{\text{proxy}} = \frac{|P_S(\varepsilon_D)|}{|P_S(X)| + \delta}
\]

For constraints:

\[
r_{\text{CONS},S}^{\text{proxy}} = \frac{|P_S(\mathcal H,\mathcal M)|}{|P_S(\partial X)|+\delta}
\]

```python
@dataclass
class MSolveReceipt:  # Attempt receipts (every attempt, accepted or not)
    attempt_id: int     # Increments on every attempt
    kappa: Kappa        # Typed kappa
    t: float            # physical time
    tau_attempt: int    # coherence time for attempts
    residual_proxy: dict  # {domain: {scale: normalized_value}} - per formula above
    dt_cap_min: float
    dominant_thread: tuple
    actions: list
    policy_hash: str
    state_hash: str       # Cheap fingerprint (rolling checksum of slices + scalars)
    stage_time: float
    stage_id: int
    sem_ok: bool          # Prerequisites initialized, no NaN/Inf
    rollback_count: int   # For attempt-level tracking
    perf: dict

@dataclass
class MStepReceipt:  # Only accepted steps (no accepted: bool needed)
    step_id: int        # Increments only on acceptance
    kappa: Kappa        # Typed kappa
    t: float            # physical time (advances only here)
    tau_step: int       # coherence time for steps
    residual_full: dict   # Full residuals per domain/scale
    enforcement_magnitude: float
    dt_used: float
    rollback_count: int  # Total rollbacks for this step
    gate_after: dict     # Last gate check outcome (should be pass)
    actions_applied: list # What rails/enforcement were used

@dataclass
class MOrchReceipt:
    o: int
    window_steps: list
    quantiles: dict  # p50/p90/p99 per domain/scale
    dominance_histogram: dict
    chatter_score: float
    regime_label: str
    promotions: list
    verification_threshold: float  # Explicit threshold used
    verification_norm: str         # Norm type used
    min_accepted_history: int      # Required history count
    policy_hash: str               # Window policy hash
```

### 4. Fill Rules & Retrieval APIs

Domain/scale-aware access:

```python
def retrieve_by_domain_scale(self, domain, scale, time_range):
    # Filter receipts by domain/scale within time range
    pass

def fill_solve_buffer(self, kappa, compute_func):
    # Ring buffer semantics - overwrite oldest
    pass

def promote_step_to_orch(self, step_receipts, gate_passed):
    # Append-only canon if gate passes
    pass
```

### 5. Pruning Logic

- **TTL Expiration**: Keep for short-term caches
- **V-Score Eviction**: For resource management
- **Canon Append-Only**: Never modify M_orch receipts
- **Regime Invalidation**: Taint by regime_hash
- **Failure Snapshot Rule**: On abort/crash, persist last K attempt receipts and policy hashes to disk for forensics

### 6. SEM Safety Mechanisms

- **Policy Fingerprinting**: Hash gauge/enforcement/filter policies per window
- **SEM Barriers**: Reject if policy_hash changes mid-window
- **Coherence Checks**: Validate time representations across κ
- **No Silent Zeros**: Prerequisites uninitialized → sem_ok=False (not eps=0)
- **No Nonfinite Residuals**: Any NaN/Inf in residuals → sem_ok=False

### 7. Integration Contracts

Extend existing contracts to use Aeonic Memory:

```python
class SolverContractWithMemory(SolverContract):
    def __init__(self, memory: AeonicMemoryContract):
        super().__init__()
        self.memory = memory

    def compute_rhs(self, X, t, gauge_policy, sources_func):
        # Emit M_solve receipt on each stage
        # Check SEM violations
        pass
```

Similar for StepperContractWithMemory and OrchestratorContractWithMemory.

### 8. New Files & Classes

- `aeonic_memory_contract.py`: Contract interface wrapper
- `phaseloom_27.py`: Full 27-thread PhaseLoom implementation
- `receipt_schemas.py`: Dataclasses for typed receipts
- `promotion_engine.py`: Logic for canon promotion and pruning
- `sem_safety.py`: Policy fingerprinting and barrier enforcement

### 9. Migration Path

1. Add new interfaces without breaking existing tier API
2. Implement PhaseLoom 27-thread lattice
3. Add detailed receipt schemas
4. Integrate SEM safety checks
5. Implement promotion contract
6. Update Solver/Stepper/Orchestrator to use new interfaces
7. Deprecate old tier API over time

### 10. Performance Considerations

- Ring buffer for M_solve: fixed size, fast overwrite
- M_step: bounded by accepted steps only
- M_orch: append-only, compressed summaries
- PhaseLoom: cheap proxy residuals at μ, full at s/o

This plan ensures full compliance with Aeonic Memory Contract v1.0 while maintaining backward compatibility.