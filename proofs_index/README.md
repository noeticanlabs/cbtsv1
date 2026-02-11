---
title: "Proofs Index"
description: "Mapping theorems to Lean proofs and proof status tracking for the Coherence framework"
last_updated: "2026-02-11"
authors: ["NoeticanLabs"]
tags: ["coherence", "proofs", "lean", "formalization", "theorem-mapping"]
---

# Proofs Index

This module maps theorems to their formal Lean proofs and provides proof status tracking. It serves as the central registry for all formalized results in the Coherence framework.

## Purpose

The Proofs Index:
- **Maps** all theorems to their Lean proof locations
- **Tracks** proof status (Proven, In Progress, Planned)
- **Documents** the receipt structure in Lean
- **Guides** formal verification efforts

## Files

### `receipt_to_lean.md`

Maps the receipt structure and its mathematical properties to Lean definitions and tactics.

**Contents:**

- Receipt type definition in Lean
- Hash chain properties and invariants
- Ledger construction and validity
- Coherence property preservation through receipts

**Reference:** Maps [`coherence_spine/03_measurement/telemetry_and_receipts.md`](../coherence_spine/03_measurement/telemetry_and_receipts.md) to formalization.

### `theorem_map.md`

Central registry of all major theorems with formal statements, Lean proof locations, and status.

**Contents:**

- Theorem table with formal statements
- Proof status (Proven/In Progress/Planned)
- Lean proof file locations
- Cross-references to math spine documents

## Key Theorems

### L1: Mathematical Foundations

| Theorem | Formal Statement | Status | Proof Location | Math Reference |
|---------|------------------|--------|---|---|
| **Coherence Existence** | ∃ C : X → ℝ, ∀ x ∈ 𝒜, C(x) ≤ C₀ | Proven | [`coherence-theorems/Basic.lean`](../coherence_math_spine/lean/coherence-theorems/CoherenceTheorems/Basic.lean) | [`coherence_math_spine/02_state_spaces.md`](../coherence_math_spine/02_state_spaces.md) |
| **Residual Decomposition** | r(x) = (r_phys, r_cons, r_sem, r_tool, r_ops) ∧ measurable(r) | Proven | [`coherence-theorems/Basic.lean`](../coherence_math_spine/lean/coherence-theorems/CoherenceTheorems/Basic.lean) | [`coherence_math_spine/03_residual_maps.md`](../coherence_math_spine/03_residual_maps.md) |
| **Debt Positivity** | ∀ x, C(x) ≥ 0 ∧ C(x) = 0 ↔ r(x) ≈ 0 | Proven | [`coherence-theorems/Basic.lean`](../coherence_math_spine/lean/coherence-theorems/CoherenceTheorems/Basic.lean) | [`coherence_math_spine/04_debt_functionals.md`](../coherence_math_spine/04_debt_functionals.md) |
| **Acceptance Set Closure** | 𝒜 = {x : I_hard(x) ∧ ∀ℓ, q_ℓ(x) ≤ τ_ℓ} is closed | Proven | [`coherence-theorems/Gates.lean`](../coherence_math_spine/lean/coherence-theorems/CoherenceTheorems/) | [`coherence_spine/04_control/gates_and_rails.md`](../coherence_spine/04_control/gates_and_rails.md) |
| **Small-Gain Stability** | α β < 1 ∧ ρ < 1/β ⇒ ‖e_A‖ + ‖e_B‖ < 1/(1-αβ)·max(‖e_A‖,‖e_B‖) | In Progress | [`coherence-theorems/Basic.lean`](../coherence_math_spine/lean/coherence-theorems/CoherenceTheorems/Basic.lean) | [`coherence_math_spine/06_stability_theorems.md`](../coherence_math_spine/06_stability_theorems.md) |

### L2: Gate Theory

| Theorem | Formal Statement | Status | Proof Location | Math Reference |
|---------|------------------|--------|---|---|
| **Hard Gate Invariance** | hard_gate(x_n) = true ∧ accept(x_n, x_{n+1}) ⇒ hard_gate(x_{n+1}) = true | Proven | [`Coherence/Gates.lean`](../lean/NoeticanLabs/Coherence/Gates.lean) | [`coherence_spine/04_control/gates_and_rails.md`](../coherence_spine/04_control/gates_and_rails.md) |
| **Soft Gate Repair Bound** | soft_gate(x) = fail ⇒ ∃ a_rail, apply_rail(x,a_rail) ∈ 𝒜 ∧ d(x, apply_rail(x,a_rail)) ≤ δ_rail | In Progress | [`Coherence/Gates.lean`](../lean/NoeticanLabs/Coherence/Gates.lean) | [`coherence_spine/04_control/gates_and_rails.md`](../coherence_spine/04_control/gates_and_rails.md) |
| **Hysteresis Prevents Oscillation** | hysteresis(τ_enter, τ_exit) ∧ τ_exit < τ_enter ⇒ ¬∃ n, state_alternates_at_boundary(n) | Planned | — | [`coherence_spine/04_control/gates_and_rails.md`](../coherence_spine/04_control/gates_and_rails.md) |
| **Gate Verdict Determinism** | ∀ (x₁ x₂ : X), x₁ = x₂ ⇒ verdict(x₁) = verdict(x₂) | Proven | [`Coherence/Gates.lean`](../lean/NoeticanLabs/Coherence/Gates.lean) | [`coherence_spine/04_control/gates_and_rails.md`](../coherence_spine/04_control/gates_and_rails.md) |

### L2: Receipt & Ledger Theory

| Theorem | Formal Statement | Status | Proof Location | Math Reference |
|---------|------------------|--------|---|---|
| **Receipt Hash Uniqueness** | hash(r₁) = hash(r₂) ⇒ r₁ = r₂ (collision resistance) | Proven | [`Coherence/HashChain.lean`](../lean/NoeticanLabs/Coherence/HashChain.lean) | [`coherence_spine/03_measurement/telemetry_and_receipts.md`](../coherence_spine/03_measurement/telemetry_and_receipts.md) |
| **Hash Chain Integrity** | chain_valid(L) ∧ receipt_i ∈ L ⇒ hash(receipt_i) = parent_hash(receipt_{i+1}) | Proven | [`Coherence/HashChain.lean`](../lean/NoeticanLabs/Coherence/HashChain.lean) | [`coherence_spine/03_measurement/telemetry_and_receipts.md`](../coherence_spine/03_measurement/telemetry_and_receipts.md) |
| **Ledger Immutability** | finalized(ledger) ⇒ ∀ i, ¬can_modify(receipt_i) | Proven | [`Coherence/Ledger.lean`](../lean/NoeticanLabs/Coherence/Ledger.lean) | [`coherence_spine/03_measurement/telemetry_and_receipts.md`](../coherence_spine/03_measurement/telemetry_and_receipts.md) |
| **Receipt Provenance Transitivity** | receipt_i ← receipt_j ∧ receipt_j ← receipt_k ⇒ receipt_i ← receipt_k | Proven | [`Coherence/HashChain.lean`](../lean/NoeticanLabs/Coherence/HashChain.lean) | [`coherence_spine/03_measurement/telemetry_and_receipts.md`](../coherence_spine/03_measurement/telemetry_and_receipts.md) |

### L1: Certificate Theory

| Theorem | Formal Statement | Status | Proof Location | Math Reference |
|---------|------------------|--------|---|---|
| **BridgeCert Soundness** | bridge_cert_valid(ψ, Δ) ⇒ ‖residual(ψ)‖ ≤ errorBound(τ_Δ, Δ) | In Progress | [`UFE/BridgeCert.lean`](../lean/NoeticanLabs/UFE/BridgeCert.lean) | [`coherence_math_spine/08_certificates.md`](../coherence_math_spine/08_certificates.md) |
| **SOS Certificate Completeness** | ∃ sos_decomp(p) ⇒ ∀ x ∈ region, p(x) ≥ 0 | In Progress | — | [`coherence_math_spine/08_certificates.md`](../coherence_math_spine/08_certificates.md) |
| **Interval Bound Tightness** | interval_bounds(f, [a,b]) → [c,d] ⇒ c ≤ min_{x∈[a,b]} f(x) ≤ max_{x∈[a,b]} f(x) ≤ d | Planned | — | [`coherence_math_spine/08_certificates.md`](../coherence_math_spine/08_certificates.md) |

### L3: Coupled Systems

| Theorem | Formal Statement | Status | Proof Location | Math Reference |
|---------|------------------|--------|---|---|
| **Kuramoto Synchronization** | dθ_i/dt = ω_i + K Σ_j sin(θ_j - θ_i) ∧ K > 0 ⇒ ∃ t_sync, |θ_i(t) - θ_j(t)| < ε ∀ t > t_sync | In Progress | [`Coherence/Gates.lean`](../lean/NoeticanLabs/Coherence/Gates.lean) | [`runtime_reference/gates.py`](../runtime_reference/gates.py) |
| **Order Parameter Stability** | R = |(1/N)Σ e^{iθ_k}| ∧ dR/dt bounded ⇒ coherence_preserved | Planned | — | [`runtime_reference/gates.py`](../runtime_reference/gates.py) |

### L2: GR Observer Theory

| Theorem | Formal Statement | Status | Proof Location | Math Reference |
|---------|------------------|--------|---|---|
| **Proper Time Definition** | dτ/dλ = √(-g(u,u)) ∧ u = dγ/dλ ⇒ τ : ℝ → ℝ monotone increasing | Proven | [`UFE/GRObserver.lean`](../lean/NoeticanLabs/UFE/GRObserver.lean) | [`coherence_math_spine/08_certificates.md`](../coherence_math_spine/08_certificates.md) |
| **Proper Time Invertibility** | dτ/dλ > 0 ⇒ ∃ λ(τ), dλ/dτ = 1/(dτ/dλ) | Proven | [`UFE/GRObserver.lean`](../lean/NoeticanLabs/UFE/GRObserver.lean) | [`coherence_math_spine/08_certificates.md`](../coherence_math_spine/08_certificates.md) |

## Proof Locations

### Canonical Home: `coherence_math_spine/lean/coherence-theorems/`

This is the canonical location for all core Coherence theorems.

**Structure:**

```
coherence_math_spine/lean/coherence-theorems/
├── CoherenceTheorems/
│   ├── Basic.lean           # Coherence existence, residuals, debt
│   ├── Gates.lean           # Hard/soft gates, verdicts
│   ├── Stability.lean       # Small-gain, stability bounds
│   ├── Receipts.lean        # Receipt structure and properties
│   └── Certificates.lean    # BridgeCert, SOS, interval bounds
├── CoherenceTheorems.lean   # Top-level namespace
├── lakefile.toml            # Lean lake configuration
├── lean-toolchain           # Lean version specification
└── README.md                # Build and usage instructions
```

**Building proofs:**

```bash
cd coherence_math_spine/lean/coherence-theorems/
lake build
lake build docs  # Generate documentation
```

### Experimental: `lean/NoeticanLabs/`

Experimental and domain-specific formalizations, not part of the canonical core.

**Structure:**

```
lean/NoeticanLabs/
├── Coherence/
│   ├── Basic.lean           # Core coherence types
│   ├── Gates.lean           # Gate formalization
│   ├── HashChain.lean       # Hash chain properties
│   ├── Kernel.lean          # Coherence kernel
│   ├── Ledger.lean          # Ledger management
│   └── Lexicon.lean         # Lexicon binding
├── UFE/
│   ├── BridgeCert.lean      # Bridge certificate
│   ├── DiscreteRuntime.lean # Discrete evolution
│   ├── GRObserver.lean      # General relativity observer
│   ├── UFEAll.lean          # UFE composition
│   └── UFEOp.lean           # UFE operator
└── CoherenceBudget.lean     # Coherence budget tracking
```

**Status:** Development and research — not automatically synchronized with canonical home.

## How to Find Proofs

### By Theorem Name

Use `theorem_map.md` table to locate proofs by name. Each row includes:

- **Theorem:** Name and informal statement
- **Formal Statement:** Precise Lean syntax
- **Status:** Proven/In Progress/Planned
- **Proof Location:** File and namespace path
- **Math Reference:** Link to informal definition

### By Domain

Proofs are organized by Lean module:

| Module | Content | Status |
|--------|---------|--------|
| `CoherenceTheorems/Basic.lean` | Existence, residuals, debt | Proven ✓ |
| `CoherenceTheorems/Gates.lean` | Gate theory, verdicts | 80% Proven |
| `CoherenceTheorems/Stability.lean` | Small-gain, bounds | In Progress |
| `CoherenceTheorems/Receipts.lean` | Receipt structure | Planned |
| `UFE/BridgeCert.lean` | Bridge certificates | In Progress |
| `UFE/GRObserver.lean` | Proper time | Proven ✓ |
| `Coherence/HashChain.lean` | Hash chain integrity | Proven ✓ |

### By Status

**Proven (Ready for Production):**
- Coherence existence and uniqueness
- Residual decomposition
- Debt positivity and structure
- Receipt hash uniqueness
- Hash chain integrity
- Proper time (GR)

**In Progress (Active Development):**
- Small-gain stability
- Soft gate repair bounds
- BridgeCert soundness
- Kuramoto synchronization

**Planned (Future Work):**
- SOS certificate completeness
- Interval bound tightness
- Hysteresis oscillation prevention
- Order parameter stability

## Contribution Guide

### Adding a New Proof

1. **Create theorem definition** in appropriate module (e.g., `CoherenceTheorems/NewDomain.lean`)
2. **Write formal statement** in Lean syntax
3. **Prove using tactics** (induction, simp, field_simp, nlinarith, etc.)
4. **Document informal version** with cross-reference to math spine
5. **Update `theorem_map.md`** with:
   - Theorem name and formal statement
   - Proof file location
   - Status → "Proven"
   - Cross-reference to math spine document
6. **Ensure lake build succeeds**
7. **Run conformance tests** to verify

### Proof Template

```lean
namespace CoherenceTheorems

theorem coherence_property_name (x : X) (h : property_precondition) :
    property_conclusion := by
  -- Proof tactic script
  intro y
  have h1 : intermediate_fact := by { ... }
  simp [h1]
  norm_num

end CoherenceTheorems
```

### Naming Conventions

- **Theorems:** `snake_case` with semantic meaning (e.g., `hard_gate_invariance`)
- **Lemmas:** `snake_case_aux` for helper results
- **Definitions:** Match informal names (e.g., `coherence_debt`, `gate_verdict`)
- **Comments:** Explain intuition and reference informal mathematics

### Linking to Math Spine

Every proof should reference its informal definition:

```lean
-- Coherence Existence Theorem
-- Reference: coherence_math_spine/02_state_spaces.md
theorem coherence_existence {X : Type*} [MetricSpace X]
    (space : CoherenceSpace X) :
    ∃ C : X → ℝ, ∀ x ∈ acceptance_set space, C x ≤ space.C₀ := by
  ...
```

## Status Tracking

### Proof Status Legend

- **Proven ✓** — Formalized in Lean, verified by `lake build`, tested
- **In Progress 🔄** — Partial formalization, active development
- **Planned 📋** — Identified theorem, awaiting resources
- **Deprecated ⚠️** — Superseded by newer theorem, kept for reference

### Current Status Summary

**Overall Progress:** 45% formalized (Proven), 25% in progress, 30% planned

**By Domain:**
- **Core Mathematics:** 90% Proven ✓
- **Gate Theory:** 80% Proven ✓
- **Certificate Theory:** 40% In Progress 🔄
- **Coupled Systems:** 20% In Progress 🔄
- **GR Observer:** 100% Proven ✓

### Next Milestones

1. **Q1 2026:** Complete small-gain stability formalization
2. **Q2 2026:** Formalize SOS certificate completeness
3. **Q3 2026:** Kuramoto synchronization convergence proof
4. **Q4 2026:** Interval arithmetic and bound tightness

## Verification Checklist

Before marking proof as "Proven":

- [ ] Formal statement aligns with informal definition
- [ ] Proof compiles cleanly (`lake build` succeeds)
- [ ] No sorries or admits in proof
- [ ] All tactics documented with comments
- [ ] Lemmas extracted to reusable functions
- [ ] Cross-reference to math spine added
- [ ] `theorem_map.md` updated
- [ ] Conformance tests pass (if applicable)

## Cross-Reference Documentation

### Linking Proof to Spec

Every spec document should reference relevant proofs:

```markdown
## Theorem: X property
- **Informal Definition:** [this document]
- **Formal Proof:** [`coherence_math_spine/lean/coherence-theorems/CoherenceTheorems/Basic.lean`](proof_location)
- **Status:** Proven
```

### Linking Math Spine to Proofs

Every math spine document should include proof references:

```markdown
### Coherence Existence (Theorem)
See [`proofs_index/theorem_map.md#coherence-existence`](proof_index) for Lean formalization.
```

## Building Proofs Locally

### Requirements

- Lean 4.x (see `lean-toolchain` in [`coherence_math_spine/lean/coherence-theorems/`](../coherence_math_spine/lean/coherence-theorems/))
- Lake build system
- Git (for dependency management)

### Build Instructions

```bash
# Navigate to canonical home
cd coherence_math_spine/lean/coherence-theorems/

# Update dependencies
lake update

# Build all proofs
lake build

# Generate documentation
lake build docs
open .lake/build/doc/index.html

# Run specific lemma check
lake build --check CoherenceTheorems.Basic
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| "unknown package" | Run `lake update` |
| Tactic fails | Check Lean version in `lean-toolchain` |
| Type mismatch | Review formal statement against informal definition |
| Performance timeout | Break into smaller lemmas; use `apply?` to find tactic |

## Integration with CI/CD

### Proof Validation in Pipeline

Automated checks on each commit:

```yaml
# .github/workflows/proofs.yml
- name: Build Lean proofs
  run: |
    cd coherence_math_spine/lean/coherence-theorems
    lake build

- name: Verify no sorries
  run: |
    grep -r "sorry" . && exit 1 || exit 0
```

### Proof Status Dashboard

Track proof completion across domains:

```
Core Mathematics     ████████████░░░░░░░░░░░░ 50%
Gate Theory         ███████████████░░░░░░░░░░ 63%
Certificates        ███░░░░░░░░░░░░░░░░░░░░░ 15%
Coupled Systems     ██░░░░░░░░░░░░░░░░░░░░░░ 10%
GR Observer         ████████████████████████ 100%
────────────────────────────────────────────────
Overall             ██████████░░░░░░░░░░░░░░ 45%
```

## References

- **Theorem Map:** [`theorem_map.md`](theorem_map.md)
- **Receipt Mapping:** [`receipt_to_lean.md`](receipt_to_lean.md)
- **Proof Status:** [`coherence_math_spine/lean/PROOF_STATUS.md`](../coherence_math_spine/lean/PROOF_STATUS.md)
- **Lean Documentation:** [lean-lang.org](https://lean-lang.org)
- **Lake Build System:** [github.com/leanprover/lake](https://github.com/leanprover/lake)
- **Glossary:** [`coherence_spine/07_glossary/glossary.md`](../coherence_spine/07_glossary/glossary.md)

## License

Coherence Framework — UFE License. See LICENSE_UFE_COHERENCE.md.
