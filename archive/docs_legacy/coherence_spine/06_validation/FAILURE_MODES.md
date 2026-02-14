---
title: "Failure Mode Taxonomy"
description: "Formal classification of failure modes, their detection, and recovery procedures"
last_updated: "2026-02-10"
authors: ["NoeticanLabs"]
tags: ["coherence", "failure", "recovery", "taxonomy", "error-handling"]
---

# Failure Mode Taxonomy

This document provides a formal classification of failure modes in coherence-governed systems, their detection mechanisms, and recovery procedures.

---

## 1. Failure Mode Categories

| Category | Code | Description |
|----------|------|-------------|
| **Numerical Instability** | N | Failures related to numerical computation |
| **Semantic Incoherence** | S | Failures related to meaning/interpretation |
| **Lexicon Violation** | L | Failures related to layer restrictions |
| **Governance Breach** | G | Failures related to policy enforcement |
| **System Failure** | X | External/hardware failures |

---

## 2. Numerical Instability (N)

### 2.1 N1: Catastrophic Divergence

**Description:** State values grow without bound, typically due to unstable discretization.

**Detection:**
- Residual norm exceeds threshold by factor > 100
- State values exceed declared bounds
- NaN or Inf detected

**Receipt Field:** `residuals.residual_norm > 100 * threshold`

**Recovery:**
1. **Immediate abort** - cannot recover by retry
2. **Reduce timestep** - restart with smaller Δt
3. **Check stability** - review discretization scheme

**Gate Trigger:** Hard gate failure

```python
if np.any(np.isnan(u)) or np.any(np.isinf(u)):
    raise FailureMode("N1: Catastrophic divergence")
```

### 2.2 N2: Stagnation

**Description:** Evolution stops or slows excessively (derivative ≈ 0 when it shouldn't be).

**Detection:**
- Residual norm ≈ 0 for extended period
- State change per step < machine epsilon

**Receipt Field:** `timing_info.steps_stagnant > 10`

**Recovery:**
1. **Check RHS** - verify drive operators are active
2. **Review constraints** - ensure no hard constraints are saturated
3. **Adjust threshold** - if stagnation is legitimate

**Gate Trigger:** Soft gate (monitor only)

### 2.3 N3: Oscillatory Instability

**Description:** State oscillates with growing amplitude.

**Detection:**
- Residual alternates sign consecutively
- Energy-like quantity increases over period

**Receipt Field:** `spectral_diagnostics.growing_mode_detected = true`

**Recovery:**
1. **Add numerical dissipation** - use upwind differencing
2. **Reduce timestep** - improve temporal resolution
3. **Apply damping rail** - reduce high-frequency components

**Gate Trigger:** Soft gate failure

### 2.4 N4: Roundoff Error Accumulation

**Description:** Loss of precision due to repeated floating-point operations.

**Detection:**
- Residual norm approaches machine epsilon
- Relative change in state < 1e-15

**Receipt Field:** `state_summary.precision_remaining < 1e-13`

**Recovery:**
1. **Checkpoint and restart** - from earlier state
2. **Increase precision** - use higher-precision arithmetic
3. **Accept limitation** - if within acceptable tolerance

**Gate Trigger:** Monitor only (informational)

---

## 3. Semantic Incoherence (S)

### 3.1 S1: Constitutive Violation

**Description:** State violates constitutive relations (e.g., density < 0 for physical system).

**Detection:**
- State violates physical constraints
- `hard_invariants` check fails

**Receipt Field:** `hard_invariants.violated = ["density_positivity"]`

**Recovery:**
1. **Apply projection rail** - project to feasible set
2. **Reject step** - do not advance state
3. **Log violation** - for audit trail

**Gate Trigger:** Hard gate failure

```python
if np.any(u < 0):
    raise FailureMode("S1: Density negativity violation")
```

### 3.2 S2: Unphysical Dynamics

**Description:** Evolution violates known physical laws (e.g., energy increase in isolated system).

**Detection:**
- Conserved quantity changes beyond tolerance
- Second law violation detected

**Receipt Field:** `conservation_drift.energy > tolerance`

**Recovery:**
1. **Review physics operator** - check Lphys implementation
2. **Verify timestep** - ensure Courant condition
3. **Apply conservation rail** - enforce constraint

**Gate Trigger:** Soft gate failure

### 3.3 S3: Semantic Drift

**Description:** Meaning of state diverges from intended interpretation.

**Detection:**
- Lexicon terms no longer apply
- Semantic embedding shifts beyond threshold

**Receipt Field:** `semantic_drift.embedding_change > threshold`

**Recovery:**
1. **Re-lexiconize** - map state to current lexicon
2. **Reset semantics** - restart from canonical state
3. **Flag for review** - human evaluation required

**Gate Trigger:** Monitor only

---

## 4. Lexicon Violation (L)

### 4.1 L1: Layer Projection Error

**Description:** Attempted projection between layers without explicit theorem.

**Detection:**
- `layer_projection` without corresponding `BridgeCert`
- Missing receipt for inter-layer operation

**Receipt Field:** `layer_violation.prohibited_projection = true`

**Recovery:**
1. **Block operation** - prevent execution
2. **Request BridgeCert** - require certified projection
3. **Log violation** - for audit

**Gate Trigger:** Hard gate failure

### 4.2 L2: Terminology Misuse

**Description:** Use of terms inconsistent with Lexicon Canon definitions.

**Detection:**
- Term appears in wrong context
- Required qualifier missing

**Receipt Field:** `terminology_violation.term = "coherence"`

**Recovery:**
1. **Auto-correct** - replace with canonical term
2. **Flag for review** - human evaluation
3. **Update lexicon** - if term needs redefinition

**Gate Trigger:** Soft gate (warning)

### 4.3 L3: Missing Attribution

**Description:** Derived work lacks proper attribution to original canon.

**Detection:**
- Receipt missing required attribution fields
- Distribution without license notice

**Receipt Field:** `attribution.missing = ["canon_reference"]`

**Recovery:**
1. **Reject distribution** - until attribution added
2. **Generate attribution** - from receipt metadata
3. **Log violation** - for compliance tracking

**Gate Trigger:** Hard gate for distribution

---

## 5. Governance Breach (G)

### 5.1 G1: Unauthorized Action

**Description:** Action taken without proper gate approval.

**Detection:**
- State change without corresponding "pass" receipt
- Timestamp gap in receipt chain

**Receipt Field:** `governance.gap_detected = true`

**Recovery:**
1. **Rollback** - revert to last approved state
2. **Audit trail** - document the breach
3. **Escalate** - notify governance authority

**Gate Trigger:** Critical failure (halt system)

### 5.2 G2: Policy Violation

**Description:** Action violates declared policy constraints.

**Detection:**
- Action exceeds declared bounds
- Policy threshold exceeded

**Receipt Field:** `policy_violation.constraint = "max_action"`

**Recovery:**
1. **Clamp action** - to policy bounds
2. **Retry** - with constrained action
3. **Escalate** - if policy too restrictive

**Gate Trigger:** Hard gate failure

### 5.3 G3: Certificate Revocation

**Description:** Action relies on a BridgeCert that has been revoked.

**Detection:**
- `bridge_cert_id` references revoked certificate
- Certificate expiration date passed

**Receipt Field:** `certificate.status = "revoked"`

**Recovery:**
1. **Reject action** - immediately
2. **Request new certificate** - from authority
3. **Document** - for compliance

**Gate Trigger:** Hard gate failure

---

## 6. System Failure (X)

### 6.1 X1: Hardware Failure

**Description:** Infrastructure failure (memory, disk, network).

**Detection:**
- Exception not caught by application code
- System call failure

**Receipt Field:** `system.error_type = "OOM" | "IOError" | "NetworkError"`

**Recovery:**
1. **Checkpoint** - save state before failure
2. **Restart** - from last valid state
3. **Notify** - system administrator

**Gate Trigger:** System-level abort

### 6.2 X2: Resource Exhaustion

**Description:** System resources (memory, CPU, disk) depleted.

**Detection:**
- Memory usage > 90%
- Queue depth > capacity
- Timeout during operation

**Receipt Field:** `resource_usage.memory_percent > 90`

**Recovery:**
1. **Throttle** - reduce computational load
2. **Garbage collect** - free unused resources
3. **Scale** - add resources if possible

**Gate Trigger:** Soft gate (degrade mode)

---

## 7. Failure Mode Reference Table

| Code | Category | Severity | Gate | Recovery |
|------|----------|----------|------|----------|
| N1 | Catastrophic Divergence | Critical | Hard | Abort, restart |
| N2 | Stagnation | Low | Monitor | Check RHS |
| N3 | Oscillatory Instability | Medium | Soft | Add dissipation |
| N4 | Roundoff Accumulation | Low | Monitor | Checkpoint |
| S1 | Constitutive Violation | Critical | Hard | Project to feasible |
| S2 | Unphysical Dynamics | Medium | Soft | Enforce conservation |
| S3 | Semantic Drift | Low | Monitor | Re-lexiconize |
| L1 | Layer Projection Error | Critical | Hard | Request BridgeCert |
| L2 | Terminology Misuse | Low | Soft | Auto-correct |
| L3 | Missing Attribution | Medium | Hard | Add attribution |
| G1 | Unauthorized Action | Critical | Hard | Rollback, audit |
| G2 | Policy Violation | Medium | Hard | Clamp, retry |
| G3 | Certificate Revocation | Critical | Hard | Request new cert |
| X1 | Hardware Failure | Critical | System | Checkpoint, restart |
| X2 | Resource Exhaustion | Medium | Soft | Throttle, scale |

---

## 8. Recovery Actions Reference

| Action | Description | When to Use |
|--------|-------------|-------------|
| **Abort** | Stop execution immediately | Critical failures |
| **Rollback** | Return to previous valid state | Governance breach |
| **Restart** | Begin from initial state | Unrecoverable failure |
| **Retry** | Attempt same action again | Transient failures |
| **Rail** | Apply correction action | Coherence repair |
| **Throttle** | Reduce computational load | Resource exhaustion |
| **Escalate** | Notify higher authority | Policy violations |

---

## 9. Integration with Coherence Debt

Each failure mode contributes to the **coherence debt**:

\[
\mathfrak{C}_{\text{total}} = \sum_k w_k \cdot \mathfrak{C}_k
\]

Where \(\mathfrak{C}_k\) is the debt from failure mode \(k\) and \(w_k\) is its weight.

| Failure Mode | Default Weight | Rationale |
|--------------|----------------|-----------|
| N1 | 1.0 | Core stability |
| N2 | 0.1 | Performance |
| N3 | 0.5 | Accuracy |
| S1 | 1.0 | Physical validity |
| G1 | 2.0 | Governance critical |
| X1 | 0.5 | Infrastructure |
| Others | 0.25 | Standard |

When \(\mathfrak{C}_{\text{total}}\) exceeds budget, the system must enter recovery mode.
