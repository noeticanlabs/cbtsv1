# Coherence Thesis: A Unified Framework for Geometric and Numerical Consistency in 3+1 General Relativity

## Executive Summary

This document presents a comprehensive theory of **coherence** as the foundational organizing principle for the CBTSV1 General Relativity solver system. Coherence unifies geometric constraints, numerical stability, temporal consistency, and auditability into a single theoretical framework. The thesis synthesizes evidence from contract specifications, implementation code, test results, and mathematical proofs to establish coherence as both a practical engineering requirement and a deep mathematical property of the gravitational field equations.

> **Research Program Status**: This document describes an ongoing research program. Claims marked [HEURISTIC] represent working hypotheses. Claims marked [OBLIGATION] represent proof obligations or ledger items pending verification.

---

## 1. Foundations: The Nature of Coherence

### 1.1 Definition and Scope

**Coherence** in the CBTSV1 context refers to the property that ensures all aspects of a numerical relativity simulation—geometric fields, constraint violations, temporal evolution, and computational state—remain mutually consistent and mathematically well-posed. Coherence is not a single metric but a multi-dimensional invariant that must be maintained across all scales of the simulation.

The coherence framework addresses three fundamental challenges in numerical relativity:

1. **Constraint Preservation**: The Hamiltonian and momentum constraints must remain satisfied (or bounded) throughout evolution
2. **Gauge Stability**: The coordinate conditions must remain well-behaved without coordinate singularities
3. **Temporal Consistency**: The discrete time evolution must accurately represent the continuous underlying PDE system

### 1.2 The Law of Coherence (LoC)

The **Law of Coherence** is the formal specification that all accepted simulation steps must satisfy:

```
LoC-PRINCIPLE-v1.0: A step is accepted iff:
  1. All geometric constraints are satisfied post-step
  2. All temporal invariants are preserved
  3. All gauge conditions remain within prescribed bounds
  4. The audit hash chain remains intact
```

The LoC is enforced through a series of **coherence gates** that validate different aspects of the simulation state before acceptance.

---

## 2. Coherence Architecture

### 2.1 Temporal Coherence System

The temporal coherence system distinguishes between two types of time:

| Time Type | Symbol | Purpose | Behavior |
|-----------|--------|---------|----------|
| Physical Time | t | Coordinate time in PDE evolution | Advances on accepted steps, rolls back on rejected steps |
| Audit/Coherence Time | τ | Monotone audit time tracking coherence | Advances only on accepted steps, remains rollback-safe |

**Physical Time (t)**: The coordinate time in PDE evolution, representing the physical progression of the system state. Advances on accepted steps but rolls back on rejected attempts.

**Audit/Coherence Time (τ)**: A monotone audit time that tracks coherence and verification, advancing only on accepted steps and remaining rollback-safe. Serves as the immutable audit trail.

### 2.2 Coherence Gates

The stepper contract implements three primary coherence gates:

| Gate | Purpose | Validation Criteria |
|------|---------|-------------------|
| **SEM** | Semantic coherence | Energy identity, constraint residuals |
| **CONS** | Constraint preservation | Hamiltonian/momentum constraint bounds |
| **PHY** | Physical validity | Gauge conditions, singularity avoidance |

Each gate corresponds to a specific invariant that must be satisfied:

```python
# From stepper_contract.md
A step is accepted iff:
1. All coherence constraints are satisfied post-step 
   (ε_H < rails_policy['eps_H_max'], ε_M < rails_policy['eps_M_max'])
2. No dt-dependent violations occur (CFL stability, stiffness checks)
```

### 2.3 Gate Unification Mapping

The CBTSV1 system unifies multiple gate taxonomies through explicit mapping:

| SEM/CONS/PHY Gate | ε-Family Residual | Policy Type | Hard/Soft |
|-------------------|-------------------|-------------|-----------|
| SEM → Energy | ε_energy (implicit) | Hard | Hard |
| CONS → Hamiltonian | ε_H | eps_H_max | Hard (CONSTRAINT) |
| CONS → Momentum | ε_M | eps_M_max | Soft (STATE) |
| PHY → Projection | ε_proj | eps_proj_max | Soft (STATE) |
| PHY → Clock | ε_clk | eps_clk_max | Soft (RATE) |

The ε-family residuals (ε_H, ε_M, ε_proj, ε_clk) are the canonical coherence metrics used throughout the implementation.

### 2.4 Stage Coherence Invariant

The **stage coherence invariant** (`N:INV.clock.stage_coherence`) ensures that all sub-operators in a step evaluate at consistent stage times:

```
N:INV.clock.stage_coherence:
  semantics: "All sub-operators in a step must evaluate at consistent 
              stage times; mismatches are treated as coherence failure."
  acceptance_rule: "metric:delta_stage_t_max == 0"
```

This invariant is critical for multi-rate integration schemes where different field components may evolve at different time resolutions.

### 2.5 Gate Types

- **Gate_step**: Per-step coherence validation (3 barriers: SEM, CONS, PHY)
- **Gate_orch**: Window-level regime stability (chatter, residual thresholds)

---

## 3. PhaseLoom: The Coherence Governance Layer

### 3.1 PhaseLoom Architecture

The PhaseLoom system serves as the "authoritative governance layer for CBTSV1, maintaining coherence across all simulation scales." It implements:

1. **Dyadic Band Analysis**: Decomposes signals into octave bands for multi-scale coherence assessment
2. **Band Coherence Computation**: Calculates coherence values C_o for each band using the **Kuramoto-style order parameter**:

   ```
   C_o = |⟨e^{i·θ_o(t)}⟩|
   
   where θ_o(t) = ∫ ω_o(t') dt' is the integrated phase of band o
   and ω_o(t) is the dyadic band-filtered signal
   ```

   > **Implementation Note**: The thesis previously showed `C_o = <ω_o>/σ(ω_o)`, which was incorrect. The actual implementation uses the Kuramoto order parameter (phaseloom_octaves.py lines 66-70).

3. **Tail Danger Assessment**: Evaluates risk of constraint violation based on spectral distribution
4. **Window-Level Regime Stability**: Monitors chatter and residual thresholds across simulation windows

### 3.2 PhaseLoom Functions

```python
# From phaseloom_octaves.py
compute_dyadic_bands()        # Compute dyadic moving-average differences
compute_band_coherence()      # Compute coherence C_o using Kuramoto order parameter
compute_tail_danger()         # Compute tail danger D_o
```

### 3.3 Band Coherence Metrics

The PhaseLoom computes the following per-band metrics:

| Metric | Symbol | Purpose |
|--------|--------|---------|
| Octave Rate | ω^{(o)} | Characteristic frequency of band o |
| Band Coherence | C_o | Kuramoto order parameter in band o (0 ≤ C_o ≤ 1) |
| Tail Danger | D_o | Risk indicator for band o |

These metrics feed into the window-level regime classification:

- **Stable**: Low residuals, coherent bands
- **Caution**: Elevated residuals, partial band coherence
- **Critical**: High residuals, incoherent bands

### 3.4 PhaseLoom Octave Math Warning

> **⚠️ HISTORY LENGTH REQUIREMENT**: Dyadic band computation requires `history_length ≥ 2^(O_max+1)` for full band resolution. The implementation sets `window_size = 2^(o+1)` for band o (line 50-51 in phaseloom_octaves.py). Insufficient history length results in zero-filled bands for high octaves.

---

## 4. Notation Ledger

### 4.1 State and Norms

| Notation | Meaning | Notes |
|----------|---------|-------|
| Ψ | State vector | Collective field variables (γ_ij, K_ij, α, β^i) |
| ‖·‖_{L²} | L² norm | Spatial L² norm of tensor fields |
| ‖·‖_{H^s} | Sobolev H^s norm | s-th derivative Sobolev norm |

### 4.2 Residual Notation

| Notation | Meaning | Units |
|----------|---------|-------|
| ε_H | Hamiltonian constraint residual | dimensionless (normalized) |
| ε_M | Momentum constraint residual | dimensionless (normalized) |
| ε_proj | Projection error residual | dimensionless |
| ε_clk | Clock synchronization residual | seconds (normalized) |

### 4.3 Determinism Assumptions

For audit/replay reproducibility, the following assumptions must hold:

| Assumption | Description | Verification Method |
|------------|-------------|---------------------|
| **D1: Pure Function** | Step function depends only on state and parameters | Static analysis |
| **D2: Deterministic Ops** | All numerical operations are deterministic | Floating-point audit |
| **D3: Fixed Seed** | Random number generators use fixed seeds | Seed registration |
| **D4: Order-Preserved** | Evaluation order is deterministic | DAG validation |
| **D5: Immutable Inputs** | Input data cannot change during step | Hash verification |

> **Reproducibility Note**: Hash chain guarantees **integrity + auditability**. Bit-for-bit reproducibility requires additional guarantees (D1-D5) and is not automatically guaranteed.

---

## 5. Coherence in Field Evolution

### 5.1 The Lambda-Damping Correction

The coherence operator implements a **constraint damping scheme** using the parameter λ (lambda):

```python
# From gr_coherence.py
def compute_dominance(self, rhs_gamma_sym6, rhs_K_sym6, rhs_phi, ...):
    B_norm = (np.linalg.norm(rhs_gamma_sym6) + np.linalg.norm(rhs_K_sym6) + ...)
    K_norm = abs(self.lambda_val) * B_norm
    eps = 1e-10
    D_lambda = abs(self.lambda_val) * K_norm / (B_norm + eps)
    return D_lambda

def apply_damping(self, fields):
    if self.damping_enabled:
        decay_factor = np.exp(-self.lambda_val * dt)  # With time step
        fields.K_sym6 *= decay_factor
```

### 5.2 Damping Units Ledger

| Symbol | Quantity | Units | Status |
|--------|----------|-------|--------|
| λ | Damping coefficient | **1/time** (s⁻¹) | Derived from stability analysis |
| Δt | Time step | seconds | Variable per step |
| λ·Δt | Damping argument | dimensionless | Argument of exponential |
| decay_factor | e^{-λ·Δt} | dimensionless | Applied to K_sym6 |

> **Damping Description**: The λ-damping term drives the extrinsic curvature K toward constraint-satisfying configurations by exponential decay. This is an **empirical stabilization scheme** derived from constraint damping schemes in numerical relativity (e.g., the Generalized Harmonic Gauge constraint transport), not a first-principles physical effect. The decay rate λ is calibrated against test problems (MMS, gauge pulses).

### 5.3 Projection-Based Coherence

The coherence operator also implements projection methods to enforce geometric constraints:

```python
def apply_projection(self, fields):
    # Enforce det(gamma_tilde) = 1
    gamma_tilde_mat = sym6_to_mat33(fields.gamma_tilde_sym6)
    det = np.linalg.det(gamma_tilde_mat)
    det_correction = det ** (-1.0/3.0)
    gamma_tilde_new = gamma_tilde_mat * det_correction[..., np.newaxis, np.newaxis]
    K_proj = mat33_to_sym6(gamma_tilde_new) - fields.gamma_tilde_sym6
    return K_proj
```

### 5.4 Coherence Budget

The system treats stability margin as a spendable "resource" through the **K-resource / Coherence Budget** mechanism:

```
Coherence Budget = f(residual_margins, gauge_stability, constraint_satisfaction)
```

When the coherence budget is depleted, the system triggers:
1. Adaptive time-step reduction
2. Operator re-evaluation
3. Potential rollback to last checkpoint

---

## 6. Coherence in Test Results

### 6.1 GCAT Test Suite Coherence Metrics

The GCAT test suite validates coherence across multiple scenarios:

| Test | Coherence Focus | Result Metric |
|------|----------------|---------------|
| GCAT1b | Overlap arbitration | Constraint residuals during field overlap |
| GCAT1c | Manifold violation | Coherence under coordinate singularities |
| GCAT2s1 | High-frequency gauge pulse | Coherence preservation under rapid gauge evolution |
| GCAT2s2 | Constraint-violating perturbation | Recovery from constraint violation |

### 6.2 Sample Coherence Receipt

```json
{
  "run_id": "R-2026-01-19-0001",
  "step_id": 128,
  "event": "A:RCPT.step.accepted",
  "thread": "A:THREAD.PHY.M.R1",
  "t": "12.800000000",
  "dt": "0.010000000",
  "tau": "12.734500000",
  "dtau": "0.009820000",
  "intent_id": "N:INV.pde.div_free",
  "intent_hash": "2f3d4c5b22e4b6b70b1eacbbcd1a56f3f3f9e91a0b8d3abf25a4a3f9246e5f1b",
  "ops": ["H64:r48","H64:r53","H64:r50","H64:r49","H64:r56","H64:r61"],
  "gates": {
    "energy_identity": {"status": "pass", "margin": "0.000002100"},
    "tail_barrier": {"status": "pass", "S_j_max": "0.420000000"}
  },
  "invariants_enforced": [
    "N:INV.pde.div_free",
    "N:INV.pde.energy_nonincreasing",
    "N:INV.clock.stage_coherence",
    "N:INV.ledger.hash_chain_intact"
  ]
}
```

### 6.3 Run Summary Coherence Statistics

```
steps_total: 130
steps_accepted: 129
steps_rejected: 1
coherence_rate: 99.23%
dominant_threads: ["PHY.M.R1", "CONS.M.R0", "PHY.H.R2"]
final_ckpt: "CKPT-000128-A"
```

### 6.4 OmegaLedger Coherence Hash Chain

The OmegaLedger provides per-step Ω-receipts with coherence hash:

```python
coherence_hash_n = SHA256(serialize(Xi_n) || coherence_hash_{n-1})
```

This creates an immutable chain that verifies the **integrity + auditability** of the simulation history.

> **Reproducibility Conditions**: Full bit-for-bit reproducibility requires:
> 1. Identical binary/dynamic library versions
> 2. Identical floating-point model (e.g., IEEE-754)
> 3. Identical parallel evaluation order (or verified determinism)
> 4. Identical random number generator states

---

## 7. Mathematical Coherence Theory

### 7.1 Yang-Mills Coherence [OBLIGATION]

The Clay Mathematics Institute's Yang-Mills problem connects to coherence through:

1. **Mass Gap as Coherence Scale**: The mass gap is the coherence scale required to maintain consistent accounting of non-Abelian charge
2. **UV Coherence**: Ultraviolet finiteness requires coherence at short distances
3. **Ward Identities**: Coherence of gauge transformations encoded in Ward residuals

From `clay_theorem_ledger_yang_mills.md`:
- **LoC Gate Mapping**: UV coherence (Lemma 5.1), Ward identities (Lemma 9)
- **Uniform Bound Required**: ε_ren[A; a, μ] ≤ C a^α with explicit C > 0, α > 0

> **[OBLIGATION]**: These are proof obligations in the Clay problem ledger, not established results.

### 7.2 Navier-Stokes Coherence [HEURISTIC]

For fluid systems, coherence relates to:

```
[HEURISTIC] Claim: For ν > 0, Navier-Stokes possesses a Self-Forming Barrier 
       that guarantees global regularity.
        
For ν = 0 (Euler), this barrier is absent, allowing incoherence (singularities).
```

This demonstrates that viscosity acts as a natural coherence mechanism.

> **[HEURISTIC]**: This claim represents a research hypothesis in the Clay problem program, not a theorem.

### 7.3 Einstein Constraint Coherence

The Hamiltonian constraint:
```
H = R + K² - K_ij K^ij - 16πρ = 0
```

Must be maintained as a coherence invariant. The momentum constraint:
```
M^i = D_j (K^{ij} - γ^{ij} K) - 8πj^i = 0
```

Similarly enforces momentum conservation coherence.

### 7.4 Budget/Coherence Obligations [OBLIGATION]

From `clay_theorem_ledger_yang_mills.md`:
- **LoC Gate Mapping**: Budget stability (Lemma 5.2), gauge coherence (Ward residuals)
- **Uniform Bound Required**: λ_GF, λ_RP > λ_min > 0 uniformly; L < L_max < 1

> **[OBLIGATION]**: These are proof obligations, not established results.

---

## 8. Implementation Reference

### 8.1 Core Coherence Modules

| Module | Path | Purpose |
|--------|------|---------|
| `gr_coherence.py` | `src/core/gr_coherence.py` | Main coherence operator with damping and projection |
| `gr_gates.py` | `src/core/gr_gates.py` | Gate checking and residual validation |
| `gr_constraints.py` | `src/core/gr_constraints.py` | Constraint computation and enforcement |
| `gr_clock.py` | `src/core/gr_clock.py` | Temporal coherence tracking |
| `omega_ledger.py` | `src/contracts/omega_ledger.py` | Coherence hash chain maintenance |
| `phaseloom_octaves.py` | `src/phaseloom/phaseloom_octaves.py` | Band coherence computation |

### 8.2 Contract Specifications

| Contract | Coherence Role |
|----------|---------------|
| [`stepper_contract.md`](specifications/contracts/stepper_contract.md) | Defines acceptance criteria and gate semantics |
| [`phaseloom_contract.md`](specifications/contracts/phaseloom_contract.md) | Band coherence computation and regime classification |
| [`temporal_system_contract.md`](specifications/contracts/temporal_system_contract.md) | Physical vs. audit time semantics |
| [`orchestrator_contract.md`](specifications/contracts/orchestrator_contract.md) | Window-level coherence aggregation |

### 8.3 Invariant Registry

| Invariant ID | Domain | Semantics |
|--------------|--------|-----------|
| `N:INV.pde.div_free` | PDE | Divergence-free constraint |
| `N:INV.pde.energy_nonincreasing` | PDE | Energy monotonicity |
| `N:INV.clock.stage_coherence` | CLOCK | Temporal consistency |
| `N:INV.ledger.hash_chain_intact` | LEDGER | Audit integrity |

### 8.4 Clock Modes

| Mode | Meaning | Tau Behavior |
|------|---------|--------------|
| `A:CLOCK.mode.real_time` | τ := t, dτ := dt | Identity coherence clock |
| `A:CLOCK.mode.coherence_time` | τ evolves by coherence-time arbitration | Must be deterministic and witnessed |

---

## 9. Coherence Verification Checklist

For any simulation run, verify:

- [ ] **Pre-Step**: All operators use consistent stage times
- [ ] **RHS Computation**: Right-hand sides computed with current field values
- [ ] **Constraint Check**: Hamiltonian/momentum residuals within tolerance
- [ ] **Gauge Check**: Lapse and shift remain positive/finite
- [ ] **Gate Validation**: All coherence gates pass before acceptance
- [ ] **Receipt Emission**: Ω-ledger receipt with coherence hash emitted
- [ ] **Hash Chain**: Chain integrity verified against previous hash

---

## 10. Coherence in Memory System

### 10.1 AEONIC Memory Coherence

The AEONIC memory system is a tiered, receipt-driven architecture that manages:

- **Simulation state**: Current field values and history
- **Provenance tracking**: Origin of each computation
- **Coherence verification**: Validation against expected invariants

### 10.2 Receipt Types

| Receipt Type | Tier | Coherence Content |
|--------------|------|-------------------|
| Attempt | Tier 1 | τ_attempt, residual_proxy |
| Step | Tier 2 | τ_step, residual_full |
| Window | Tier 3 | Regime hashes, band metrics |

### 10.3 Coherence in Memory Contracts

From `aeonic_memory_contract_alignment_plan.md`:
- **Gap**: Insufficient detail for coherence verification
- **Spec**: Detailed schemas per memory type with stamps, residuals, actions, policy hashes

---

## 11. Conclusions

Coherence in CBTSV1 is not merely a debugging tool but a fundamental organizing principle that:

1. **Ensures Physical Validity**: Prevents coordinate and constraint violations
2. **Enables Reproducibility**: Audit hash chain guarantees integrity + auditability (see Section 4.3 for reproducibility conditions)
3. **Supports Adaptive Control**: Coherence metrics inform time-step and resolution adaptation
4. **Provides Mathematical Rigor**: Connections to Clay problem coherence theory (research program, obligations pending)

The coherence framework bridges the gap between numerical implementation and mathematical physics, providing a rigorous foundation for 3+1 General Relativity simulations.

---

## 12. Critical Implementation Notes

> **⚠️ IMPORTANT**: This section documents known issues and discrepancies between the thesis and the actual implementation. These are actively being addressed.

### 12.1 Multiple C_o Definitions (Spec Drift)

The codebase contains **conflicting definitions** of the coherence symbol `C_o`:

| Module | File | Definition | Purpose |
|--------|------|------------|---------|
| `PhaseLoomOctaves` | `phaseloom_octaves.py:66-70` | `C_o = |⟨e^{i·θ}⟩|` (Kuramoto order parameter) | Band phase alignment |
| `PhaseLoomThreadsGR` | `phaseloom_threads_gr.py:304` | `C_o = min(omega_current)` | Drop detection threshold |

> **[ACTION REQUIRED]**: These metrics serve different purposes and should be renamed:
> - `C_o` (Kuramoto) → `Z_o` (order parameter) or `alignment_o`
> - `C_o` (min) → `omega_min` or `activity_floor`

### 12.2 Octave System Disabled by Default

The `PhaseLoomOctaves` class is instantiated with:

```python
# phaseloom_octaves.py:17
def __init__(self, N_threads=27, max_octaves=8, history_length=2):
```

With `history_length=2` and window sizes `2^(o+1)`:
- o=0: window=4 → **exceeds history, returns 0**
- o=1: window=8 → **exceeds history, returns 0**
- ... all bands ≥ o=1 are hard-zeroed

> **[ACTION REQUIRED]**: Either increase `history_length ≥ 2^(O_max+1)` or redesign band extraction for online operation.

### 12.3 Phase Integration Bug

**Current implementation** (phaseloom_octaves.py:66):

```python
theta_band = np.cumsum(omega_band[:, o])  # Integrates ACROSS THREADS
```

This computes cumulative sum **across thread index**, not across time. This is not temporal phase evolution.

**Correct implementation should**:
- Track `theta_band` as a time series per thread
- Integrate `ω_o(t)` over time dimension, not thread dimension

> **[ACTION REQUIRED]**: Refactor to store temporal history per band, not thread history.

### 12.4 Impact Assessment

| Issue | Severity | Impact |
|-------|----------|--------|
| Multiple C_o | Medium | Confusion in specs vs code |
| Octaves disabled | **High** | Multiscale analysis non-functional |
| Phase bug | **High** | Coherence computation incorrect |

---

## 13. References

- [`stepper_contract.md`](specifications/contracts/stepper_contract.md) - Step acceptance criteria
- [`phaseloom_contract.md`](specifications/contracts/phaseloom_contract.md) - Band coherence computation
- [`temporal_system_contract.md`](specifications/contracts/temporal_system_contract.md) - Temporal semantics
- [`orchestrator_contract.md`](specifications/contracts/orchestrator_contract.md) - Window-level governance
- [`gr_coherence.py`](src/core/gr_coherence.py) - Coherence operator implementation
- [`gr_gates.py`](src/core/gr_gates.py) - Gate checking implementation
- [`phaseloom_octaves.py`](src/phaseloom/phaseloom_octaves.py) - **⚠️ Contains bugs (see Section 12)**
- [`phaseloom_threads_gr.py`](src/phaseloom/phaseloom_threads_gr.py) - **⚠️ Different C_o definition**
- [`omega_ledger.py`](src/contracts/omega_ledger.py) - Audit hash chain
- [`loc_clay_proof_skeleton_yang_mills.md`](specifications/theory/loc_clay_proof_skeleton_yang_mills.md) - Mathematical theory
- [`loc_clay_proof_skeleton_navier_stokes.md`](specifications/theory/loc_clay_proof_skeleton_navier_stokes.md) - Fluid coherence

---

*Document Version: 2.2*  
*Updated: 2026-01-31*  
*System: CBTSV1 GR Solver Coherence Framework*
