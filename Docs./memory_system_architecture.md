# Memory System Architecture

## Overview

The AEONIC memory system is a tiered, receipt-driven memory architecture that manages simulation state, provenance tracking, and coherence verification across the GR solver and NSC coupling layers. It integrates with the clock system for time-based TTL management and provides strong semantic guarantees through contract enforcement.

---

## 1. Memory Hierarchy

### 1.1 Global Memory Tiers

| Tier | Name | Purpose | TTL (Short/Long) | Persistence |
|------|------|---------|-----------------|-------------|
| **Tier 1** | `M_solve` | Ring buffer for solve attempts | 1h / 1d | Ephemeral |
| **Tier 2** | `M_step` | Only accepted steps | 10h / 1 week | Semi-persistent |
| **Tier 3** | `M_orch` | Canon promotions (append-only) | 30d / 1 year | Long-term |

### 1.2 Local Memory Components

| Component | Scope | Purpose |
|-----------|-------|---------|
| **[`PhaseLoomMemory`](src/phaseloom/phaseloom_memory.py:7)** | Per-orchestrator | Tracks phase loom computation state, band updates, octave culling |
| **[`GRLedger`](src/core/gr_ledger.py:7)** | Global | Persistent receipt logging to `aeonic_receipts.jsonl` |
| **[`OmegaLedger`](src/contracts/omega_ledger.py:81)** | Global | Coherence hash chain for step verification |

---

## 2. Memory Types

### 2.1 Receipt Memory

Receipts are the primary memory construct. Three receipt types correspond to memory tiers:

| Receipt Type | Tier | Schema | Keys |
|--------------|------|--------|------|
| **`MSolveReceipt`** | 1 | Attempt ID, κ=(o,s,μ), residuals, dt_cap, policy_hash | `attempt_{o}_{s}_{mu}_{attempt_id}` |
| **`MStepReceipt`** | 2 | Step ID, κ=(o,s), post-step residuals, dt_used, gate_result | `step_{o}_{s}` |
| **`MOrchReceipt`** | 3 | Orchestration o, window_steps, quantiles, regime_label, promotions | `orch_{o}` |

### 2.2 Ledger Memory

**[`GRLedger`](src/core/gr_ledger.py:7)** provides:
- Hash-chained receipts (`receipt_hash`, `prev_receipt_hash`)
- Persistent JSONL storage
- Event types: `LAYOUT_VIOLATION`, `STAGE_RHS`, `CLOCK_DECISION`, `LEDGER_EVAL`, `STEP_ACCEPT/REJECT`

**[`OmegaLedger`](src/contracts/omega_ledger.py:81)** provides:
- Per-step Ω-receipts with coherence hash: `HASH(serialize(Xi_n) || prev_hash_{n-1})`
- LoC-PRINCIPLE-v1.0 minimal schema: `delta_norm`, `eps_model`, `eta_rep`, `gamma`, `clock_tau`, `clock_margin`
- Chain verification via `verify_chain()`

### 2.3 State Memory

**[`PhaseLoomMemory`](src/phaseloom/phaseloom_memory.py:7)** manages:
- `prev_K`, `prev_gamma`: Previous curvature/connection state for delta computation
- `bands_updated`: 8-element boolean array for octave band tracking
- `dominant_band`, `amplitude`: Band metrics from phaseloom analysis
- `tainted`: Regime shift flag

---

## 3. Core Components

### 3.1 AeonicMemoryBank

**File:** [`aeonic_memory_bank.py`](src/aeonic/aeonic_memory_bank.py:27)

```python
class AeonicMemoryBank:
    tiers: Dict[int, Dict[str, Record]]  # Tier → Key → Record
    tier_bytes: Dict[int, int]            # Tier → total bytes
    total_bytes: int
```

**Record Schema:**
```python
@dataclass
class Record:
    key: str
    tier: int
    payload: Any
    bytes: int
    created_tau_s/l/m: int          # Creation timestamps (3 clock scales)
    last_use_tau_s/l: int
    ttl_s/l: int                    # Time-to-live (short/long)
    reuse_count: int
    recompute_cost_est: float
    risk_score: float
    tainted: bool
    regime_hashes: List[str]
    demoted: bool
```

**Key Methods:**
- `put(key, tier, payload, ...)`: Insert/update with TTL and risk metadata
- `get(key, tier=None)`: Retrieve with automatic tier search
- `maintenance_tick()`: TTL expiration, Tier 2→3 demotion, V-score eviction
- `invalidate_by_regime(regime_hash)`: Mark records as tainted

### 3.2 AeonicMemoryContract

**File:** [`aeonic_memory_contract.py`](src/aeonic/aeonic_memory_contract.py:12)

Enforces contract compliance between memory tiers and receipt types:

| Method | Action | Memory Tier |
|--------|--------|-------------|
| `put_attempt_receipt(kappa, receipt)` | Store attempt in ring buffer | M_solve (Tier 1) |
| `put_step_receipt(kappa, receipt)` | Store accepted step | M_step (Tier 2) |
| `put_orch_receipt(receipt)` | Canon promotion (append-only) | M_orch (Tier 3) |

**SEM Barriers:**
- `check_no_silent_zeros(state, geometry)`: Prerequisites initialized
- `check_no_nonfinite(values)`: No NaN/Inf in residuals
- `validate_policy_consistency(current, window)`: Policy hash match
- `abort_on_state_gate_no_repair(gate)`: Fast-exit on unrepairable state violations

### 3.3 AML (Aeonic Memory Language)

**File:** [`aml.py`](src/core/aml.py:49)

Legality layer enforcing NSC↔GR/NR coupling:

**Compartments:**
```python
COMPARTMENTS = ["SOLVE", "STEP", "ORCH", "S_PHY"]
```

**Coupling Matrix (allowed transitions):**
```
SOLVE → STEP → ORCH → S_PHY
```

**Transaction Support:**
- `begin_transaction(tag)`: Snapshot state (deep copy tiers, counters)
- `commit_transaction()`: Persist receipts to memory
- `rollback_transaction()`: Restore snapshot on failure

---

## 4. Contract System

### 4.1 Contract Hierarchy

| Contract | File | Purpose |
|----------|------|---------|
| **`SolverContract`** | [`gr_solver.py`](src/core/gr_solver.py:7) | Enforce stage time usage, prerequisites |
| **`StepperContract`** | [`gr_stepper.py`](src/core/gr_stepper.py:5) | Attempt/step receipt separation, rollback safety |
| **`TemporalSystemContract`** | [`temporal_system_contract.py`](src/contracts/temporal_system_contract.py:3) | Time management (t, τ, o, s, μ, attempt_id) |
| **`OrchestratorContract`** | [`orchestrator_contract.py`](src/contracts/orchestrator_contract.py:5) | Window aggregation, regime transitions, promotions |
| **`PhaseLoomContract`** | [`phaseloom_contract.py`](src/contracts/phaseloom_contract.py:5) | dt caps, action suggestions per tier (R0/R1/R2) |

---

## 5. Memory Flow During Simulation Steps

### 5.1 Step Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIMULATION STEP LOOP                          │
├─────────────────────────────────────────────────────────────────┤
│ 1. SENSE (SensePhase)                                           │
│    - Compute constraints, residuals (eps_H, eps_M)              │
│    - SEM pre-step validation                                    │
│                                                                  │
│ 2. PROPOSE (ProposePhase)                                       │
│    - Threads propose dt values                                  │
│    - Sync to PhaseLoom27 if available                           │
│                                                                  │
│ 3. DECIDE (DecidePhase)                                         │
│    - dt_commit = min(dt_target, dt_cap)                         │
│    - dominant_thread selection                                  │
│                                                                  │
│ 4. COMMIT (CommitPhase) ← STATE BACKUP HERE                     │
│    - Backup fields (gamma, K, alpha, beta, phi)                 │
│    - stepper.step() with dt_commit                              │
│    - PhaseLoom computation if step accepted                     │
│    - loom_memory.post_loom_update()                             │
│                                                                  │
│ 5. VERIFY (VerifyPhase)                                         │
│    - Recompute constraints post-step                            │
│    - SEM post-step audit                                        │
│                                                                  │
│ 6. RAIL_ENFORCE (RailEnforcePhase)                              │
│    - Gate checks via rails                                      │
│    - PhaseLoom Gate_step validation                             │
│                                                                  │
│ 7. RECEIPT (ReceiptPhase) ← PERSIST TO MEMORY                   │
│    - emit_m_step() → M_step receipt → M_step (Tier 2)           │
│    - Emit to OmegaLedger with coherence hash                    │
│                                                                  │
│ 8. RENDER (RenderPhase)                                         │
│    - Update visualization channels                              │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Receipt Emission Sequence

```
Attempt Loop (per dt attempt):
├── MSolveReceipt → M_solve (Tier 1)
│   └── attempt_counter++ (always increments)
│
└── If accepted:
    └── MStepReceipt → M_step (Tier 2)
        └── step_counter++ (only on acceptance)
        └── OmegaReceipt → OmegaLedger (coherence hash chain)
        
Orchestration Window:
├── Collect M_step receipts (only accepted)
├── MOrchReceipt → M_orch (Tier 3)
│   └── regime_label: 'stable' | 'constraint-risk' | 'semantic-risk' | 'perf-risk'
└── Promotion/Quarantine decisions
```

---

## 6. Clock Integration

### 6.1 Clock Scales

The memory system integrates with [`AeonicClockPack`](src/aeonic/aeonic_clocks.py) (3 scales):

| Scale | Symbol | Purpose | Memory TTL Reference |
|-------|--------|---------|---------------------|
| **Short** | τ_s | Attempt-level | `ttl_s` |
| **Long** | τ_l | Step-level | `ttl_l` |
| **Medium** | τ_m | Maintenance | `ttl_m` |

### 6.2 Clock-Memory Synchronization

**In [`AeonicMemoryBank.maintenance_tick()`](src/aeonic/aeonic_memory_bank.py:176):**
```python
def maintenance_tick(self):
    self.clock.tick_maintenance()  # Advance all clocks
    
    # TTL expiration uses both scales
    if (self.clock.tau_s > record.created_tau_s + record.ttl_s or
        self.clock.tau_l > record.created_tau_l + record.ttl_l):
        expire_record()
```

**In [`PhaseLoomMemory`](src/phaseloom/phaseloom_memory.py:122-159):**
- `_should_compute_with_unified_clock()`: Uses clock for regime shift detection
- `detect_regime_shift(residual_slope)`: Triggers memory tainting
- `compute_regime_hash(dt, dominant_band, resolution)`: Creates regime fingerprint

### 6.3 Regime-Based Invalidation

```
Regime Shift Detected
        ↓
PhaseLoomMemory.tainted = True
        ↓
AeonicMemoryBank.invalidate_by_regime(regime_hash)
        ↓
All records with matching regime_hash → tainted=True
```

---

## 7. Persistence and Checkpointing

### 7.1 Persistent Storage

| Storage | Format | Location | Purpose |
|---------|--------|----------|---------|
| **GRLedger** | JSONL | `aeonic_receipts.jsonl` | Full step-by-step audit trail |
| **OmegaLedger** | In-memory + verify | Chain integrity | Coherence verification |
| **Tier 3 (M_orch)** | AeonicMemoryBank | Long TTL (1 year) | Canon promotion history |

### 7.2 Checkpoint Strategy

1. **State Backup** (per step): [`CommitPhase`](src/core/phases.py:236-243)
   - Fields backup before step execution
   - Used for rollback on rejection

2. **Receipt Chain** (continuous):
   ```
   receipt_hash_n = SHA256(serialize(receipt_n) || receipt_hash_{n-1})
   ```

3. **Omega Coherence Chain** (per step):
   ```
   coherence_hash_n = SHA256(serialize(Xi_n) || coherence_hash_{n-1})
   ```

### 7.3 Recovery Points

| Recovery Type | Source | Trigger |
|---------------|--------|---------|
| **Rollback** | State backup in `CommitPhase` | Step rejection, SEM violation |
| **Regime Recovery** | Tainted records + invalidation | Regime shift detected |
| **Chain Repair** | `OmegaLedger.verify_chain()` | Verification failure |

---

## 8. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AEONIC MEMORY SYSTEM                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    AML (Legality Layer)                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐              │   │
│  │  │  SOLVE   │→ │   STEP   │→ │   ORCH   │→ │  S_PHY   │              │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘              │   │
│  │       │               │              │              │                  │   │
│  │       ▼               ▼              ▼              ▼                  │   │
│  │  ┌──────────────────────────────────────────────────────────────┐    │   │
│  │  │              AeonicMemoryContract                            │    │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │    │   │
│  │  │  │put_attempt  │  │put_step    │  │put_orch (append)    │  │    │   │
│  │  │  │→ M_solve    │  │→ M_step    │  │→ M_orch             │  │    │   │
│  │  │  │ (Tier 1)    │  │ (Tier 2)   │  │ (Tier 3)            │  │    │   │
│  │  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │    │   │
│  │  └──────────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                          │
│                                    ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    AeonicMemoryBank                                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐   │   │
│  │  │   TIER 1    │  │   TIER 2    │  │           TIER 3            │   │   │
│  │  │  M_solve    │  │  M_step     │  │         M_orch              │   │   │
│  │  │  (1h/1d)    │  │ (10h/1week) │  │        (30d/1yr)           │   │   │
│  │  │ Ring Buffer │  │  Accepted   │  │      Canon Promotions      │   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘   │   │
│  │                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │  Maintenance: TTL expiration, Demotion (T2→T3), Eviction     │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                          │
│         ┌──────────────────────────┼──────────────────────────┐              │
│         ▼                          ▼                          ▼              │
│  ┌─────────────┐          ┌─────────────┐          ┌─────────────┐           │
│  │  GRLedger   │          │ OmegaLedger │          │ PhaseLoom   │           │
│  │ (JSONL)     │          │ (Hash Chain)│          │  Memory     │           │
│  └─────────────┘          └─────────────┘          └─────────────┘           │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                         CLOCK SYSTEM INTEGRATION                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    AeonicClockPack                                    │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                            │   │
│  │  │  τ_s     │  │  τ_l     │  │  τ_m     │                            │   │
│  │  │ (Short)  │  │  (Long)  │  │ (Medium) │                            │   │
│  │  └──────────┘  └──────────┘  └──────────┘                            │   │
│  │       │              │              │                                 │   │
│  │       └──────────────┴──────────────┘                                 │   │
│  │                    │                                                 │   │
│  │                    ▼                                                 │   │
│  │  TTL tracking, regime hashing, regime shift detection                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────┤
│                         CONTRACT ENFORCEMENT                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────┐  ┌───────────┐│    │
│  │  │   Solver   │  │  Stepper   │  │    Temporal    │  │ Orchestrator│   │
│  │  │ Contract   │  │ Contract   │  │    System      │  │  Contract  │    │
│  │  └────────────┘  └────────────┘  │    Contract    │  └───────────┘    │
│  │                                └────────────────┘                     │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Key Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| [`aeonic_memory_bank.py`](src/aeonic/aeonic_memory_bank.py) | 247 | Core tiered memory storage |
| [`aeonic_memory_contract.py`](src/aeonic/aeonic_memory_contract.py) | 180 | Contract-compliant memory access |
| [`aml.py`](src/core/aml.py) | 263 | Legality layer for coupling |
| [`phaseloom_memory.py`](src/phaseloom/phaseloom_memory.py) | 321 | Phase loom state tracking |
| [`gr_ledger.py`](src/core/gr_ledger.py) | 162 | Receipt persistence |
| [`omega_ledger.py`](src/contracts/omega_ledger.py) | 235 | Coherence hash chain |
| [`phases.py`](src/core/phases.py) | 470 | Phase-based orchestrator |
| [`gr_solver.py`](src/core/gr_solver.py) | 123 | Solver contract |
| [`gr_stepper.py`](src/core/gr_stepper.py) | 75 | Stepper contract |
| [`temporal_system_contract.py`](src/contracts/temporal_system_contract.py) | 80 | Temporal contract |
| [`orchestrator_contract.py`](src/contracts/orchestrator_contract.py) | 50 | Orchestrator contract |
| [`phaseloom_contract.py`](src/contracts/phaseloom_contract.py) | 37 | Phase loom contract |

---

## 10. Test Coverage

| Test File | Coverage |
|-----------|----------|
| [`tests/test_memory_system.py`](tests/test_memory_system.py) | Basic put/get, TTL, tier management |
| [`test_aeonic_memory_contract.py`](test_aeonic_memory_contract.py) | Attempt/step separation, SEM barriers |
| [`test_aml.py`](test_aml.py) | Thread tagging, compartments, transactions |
