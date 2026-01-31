# NSC_NS Dialect Specification (v1.0-draft, repo-aligned)

**Status:** Draft - Requires implementation
**Date:** 2026-01-31
**Related:** `Technical_Data/coherence_thesis_extended_canon_v2_1.md`, `plans/nsc_gr_dialect_spec.md`

---

## 0) Executive Summary

NSC_NS is the Navier-Stokes dialect of the Noetica Symbolic Compiler (NSC), designed to integrate with the LoC-Time commit protocol, PhaseLoom coherence framework, and Ω-ledger receipt system. It inherits the global invariant structure from the coherence thesis and makes NS-specific invariants operational.

---

## 1) Clay-Scope Lock

### 1.1 Input → Claim → Output

**Input:**
- Incompressible Navier-Stokes state (discrete fields: velocity **u**, pressure **p**)
- Deterministic run policy (dt rules, projection schedule, gate thresholds, receipt schema)
- Viscosity parameter **ν > 0** (for NSE mode; ν = 0 for Euler)
- Forcing field **f(x,t)** (optional)

**Claim:**
Every **accepted** step is simultaneously:
1. **Divergence-controlled** (incompressibility maintained)
2. **Energy-accounted** (no silent creation/destruction)
3. **Time-coherent** (no stage-time smearing)
4. **Ledger-closed** (hash chain intact)

**Output:**
- Sequence of accepted steps with Ω-receipts
- Audit trail for invariant enforcement

### 1.2 Research Claim (Proof Obligation)

> "For ν>0, Navier–Stokes possesses a Self-Forming Barrier that guarantees global regularity."

**Status:** [LEMMA-NEEDED] - This is a research claim, not an established theorem.

---

## 2) Required Invariants

### 2.1 NSC_NS-Specific Invariants

| ID | Name | Meaning |
|----|------|---------|
| `N:INV.pde.div_free` | Divergence-Free | "velocity remains divergence-free within declared tolerance, under declared projection schedule" |
| `N:INV.pde.energy_nonincreasing` | Energy Conservation | "kinetic energy does not increase beyond policy margin once forcing/viscosity are accounted for" |

### 2.2 Shared Invariants (from Coherence Thesis)

| ID | Name | Meaning |
|----|------|---------|
| `INV.clock.stage_coherence` | Stage Coherence | No sub-operator in a step evaluates on mismatched stage times |
| `INV.ledger.hash_chain_intact` | Ledger Integrity | Every receipt hash commits to the prior hash |

---

## 3) Execution Model (LoC-Time Compliant)

NSC_NS implements the 7-step commit protocol:

```
OBSERVE → DECIDE → ACT-PHY → ACT-CONS → AUDIT → ACCEPT/ROLLBACK → RECEIPT
```

### 3.1 Step Details

| Phase | Description | NSC_NS Specifics |
|-------|-------------|------------------|
| **OBSERVE** | Compute candidate dt bounds + coherence metrics | CFL estimate, tail danger, energy trend |
| **DECIDE** | Single arbiter chooses dt and stage times | PhaseLoom arbitration for dt selection |
| **ACT-PHY** | Advance physics to provisional state | RK timestep on NS:RHS |
| **ACT-CONS** | Enforce constraints (projection) | Helmholtz-Hodge projection for div-free |
| **AUDIT** | Compute residuals + invariant drift | Compute ε_div, ε_E, tail metrics |
| **ACCEPT/ROLLBACK** | Accept iff gates pass | Gate evaluation with margins |
| **RECEIPT** | Emit hash-chained receipt | Full state with invariant proofs |

### 3.2 Dual Clock Semantics

- **Physical time (t):** May rollback on rejected attempts
- **Audit time (τ):** Advances only on accepted steps

---

## 4) State Model

### 4.1 Physical Fields

| Field | Type | Description |
|-------|------|-------------|
| `u` | Vector field (3D) | Velocity **u(x)** |
| `p` | Scalar field | Pressure **p(x)** (may be implicit via projection) |
| `f` | Vector field (3D) | Forcing **f(x,t)** (optional) |
| `ν` | Scalar | Viscosity **ν > 0** (required for NSE; ν = 0 for Euler) |

### 4.2 Discrete Metadata

| Field | Description |
|-------|-------------|
| `grid_id` | Grid/basis identifier (N, dx, domain) |
| `boundary_mode` | periodic / wall / inflow-outflow |
| `projection_method` | Hodge / Poisson solver ID |
| `filter_policy` | De-aliasing policy (if any) |

---

## 5) Dialect Surface Syntax

### 5.1 Primitive Glyph Intents (Phase-1)

| Glyph | Intent | Maps To |
|-------|--------|---------|
| `φ(…)` | Diffusion | ν∇² terms |
| `⊕(…)` | Source | Forcing/injection |
| `⊖(…)` | Damping | Explicit damping/filters |
| `◯(…)` | Boundary | Boundary conditions |
| `∆` | Time marker | Declares stage/commit times |
| `□(…)` | Gate intent | Enforce + verify invariants |
| `⇒` | Compile intent | Seal canonical compiled block |

### 5.2 NSC_NS Macro Layer

#### (A) NS:RHS - Physics Right-Hand Side

**Macro:** `NS:RHS(u, p, ν, f)`

Expands to:
- **Nonlinear advection:** `ADV(u) := (u·∇)u`
- **Pressure gradient:** `GRAD(p) := ∇p`
- **Diffusion:** `φ(ν∇²u)`
- **Forcing:** `⊕(f)`

#### (B) NS:PROJ - Incompressibility Projection

**Macro:** `NS:PROJ(u*) → u`

- Returns divergence-free velocity via Helmholtz-Hodge projector
- Executed in ACT-CONS phase

#### (C) NS:AUDIT_ENERGY - Energy Accounting

**Macro:** `NS:AUDIT_ENERGY(u, f, ν)`

Computes:
- Kinetic energy: `E = ½∫|u|²`
- Dissipation: `D = ν∫|∇u|²`
- Power injection: `P = ∫u·f`

---

## 6) Gates (SEM / CONS / PHY)

### 6.1 CONS Gate (Divergence-Free)

| Property | Value |
|----------|-------|
| Metric | `ε_div = ||∇·u||` |
| Rule | Pass iff `ε_div ≤ eps_div_max` |
| Action on fail | Rollback + dt shrink |

### 6.2 SEM Gate (Energy Identity)

| Property | Value |
|----------|-------|
| Metric | `ε_E = |ΔE - (Δt*P - Δt*D)|` |
| Rule | Pass iff `ε_E ≤ eps_E_max` |
| Interpretation | "No silent energy creation" firewall |

### 6.3 PHY Gate (Physical Validity)

| Property | Value |
|----------|-------|
| Checks | No NaNs/Infs |
| | ν ≥ ν_min > 0 |
| | CFL/stability bounds |
| | Boundary sanity |

---

## 7) Receipt Schema (Ω-Ledger)

NSC_NS receipts follow the structure from the coherence thesis:

```json
{
  "run_id": "uuid",
  "step_id": "int",
  "hash_prev": "sha256",
  "hash_curr": "sha256",
  "times": {
    "t": 0.0,
    "dt": 0.001,
    "tau": 0.0,
    "dtau": 0.001
  },
  "stage_times": {
    "observe": 0.0,
    "decide": 0.0,
    "act_phy": 0.0,
    "act_cons": 0.0,
    "audit": 0.0
  },
  "ops": [
    {"op": "NS:RHS", "params": {...}},
    {"op": "NS:PROJ", "params": {...}},
    {"op": "NS:AUDIT_ENERGY", "params": {...}}
  ],
  "gates": {
    "CONS": {"passed": true, "margin": 1e-6, "value": 1e-8},
    "SEM": {"passed": true, "margin": 1e-4, "value": 1e-6},
    "PHY": {"passed": true, "reason": "all checks passed"}
  },
  "invariants_enforced": [
    "N:INV.pde.div_free",
    "N:INV.pde.energy_nonincreasing",
    "INV.clock.stage_coherence",
    "INV.ledger.hash_chain_intact"
  ],
  "metrics": {
    "E": 0.5,
    "D": 0.01,
    "P": 0.0,
    "tail_danger": 0.001
  }
}
```

---

## 8) Proof Obligations (Gap Budget ≤ 3)

| # | Obligation | Status | Description |
|---|------------|--------|-------------|
| 1 | Self-Forming Barrier Lemma | [LEMMA-NEEDED] | Prove NS regularity for ν>0 |
| 2 | Projection Error Bound | [LEMMA-NEEDED] | Show projector doesn't inject energy beyond SEM margins |
| 3 | Tail Danger Metric (NS) | [LEMMA-NEEDED] | Define concrete signal (u? ω? residual?) and threshold policy |

---

## 9) Red-Team Tests

### 9.1 Zero Solution Test

**Input:** u=0, f=0
**Expected:** Gates trivially pass, energy stays 0
**Status:** [PROVED-as-check]

### 9.2 High-Frequency Initial Data

**Input:** Random/oscillatory initial velocity
**Expected:** Tail danger spikes, dt shrinks, receipts show dominance
**Status:** [HEURISTIC until implemented]

### 9.3 ν→0 Limit (Euler Mode)

**Input:** ν approaching 0
**Expected:** Switch to Euler mode, do not claim self-forming barrier
**Status:** [PROVED-as-policy]

---

## 10) Pseudo-Dialect Sketch (Single Step)

```nllc
// NSC_NS Step Declaration
fn ns_step(u: Field<Vector3>, p: Field<Scalar>, 
           ν: Float, f: Field<Vector3>,
           dt: Float) -> (Field<Vector3>, Field<Scalar>, Receipt) {
    
    // OBSERVE: Compute metrics
    let tail_danger = phase_loom_tail_danger(u);
    let cfl = compute_cfl(u, dt);
    
    // DECIDE: Get dt from PhaseLoom
    let dt_chosen = phase_loom_arbitrate(dt, tail_danger, cfl);
    
    // ACT-PHY: RK timestep on NS:RHS
    let u_star = rk_step(u, p, ν, f, dt_chosen, NS:RHS);
    
    // ACT-CONS: Project to divergence-free
    let u_divfree = NS:PROJ(u_star);
    
    // AUDIT: Compute metrics and invariant drift
    let (ε_div, ε_E, E, D, P) = audit_ns(u_divfree, f, ν);
    
    // GATE: Evaluate gates
    let (cons_pass, cons_margin) = gate_cons(ε_div);
    let (sem_pass, sem_margin) = gate_sem(ε_E);
    let (phy_pass, phy_reason) = gate_phy(u_divfree, ν);
    
    // ACCEPT/ROLLBACK
    if cons_pass && sem_pass && phy_pass {
        let receipt = emit_receipt(
            step_id, hash_prev, 
            {t, dt_chosen, tau, dtau},
            [NS:RHS, NS:PROJ, NS:AUDIT_ENERGY],
            {CONS: cons_pass, SEM: sem_pass, PHY: phy_pass},
            {E, D, P, tail_danger}
        );
        return (u_divfree, p, receipt);
    } else {
        return (u, p, null_receipt);  // Rollback
    }
}
```

---

## 11) Implementation Roadmap

| Phase | Task | Status |
|-------|------|--------|
| 1 | Define NS type system (VectorField, ScalarField) | Not started |
| 2 | Implement NS:RHS intrinsic | Not started |
| 3 | Implement NS:PROJ (Hodge projector) | Not started |
| 4 | Implement NS:AUDIT_ENERGY | Not started |
| 5 | Define gate thresholds (eps_div_max, eps_E_max) | Not started |
| 6 | Create NS-specific receipt schema | Not started |
| 7 | Implement red-team tests | Not started |
| 8 | Integrate with PhaseLoom tail danger | Not started |

---

## 12) References

- `Technical_Data/coherence_thesis_extended_canon_v2_1.md` - Core coherence theory
- `plans/nsc_gr_dialect_spec.md` - GR dialect reference
- `specifications/contracts/phaseloom_contract.md` - PhaseLoom integration
- `specifications/contracts/orchestrator_contract.md` - Commit protocol
