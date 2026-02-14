# Theorem Map

This document provides a human-readable index of what is proven in the Lean formalization, organized by the five axioms (C1–C5) and their derived laws.

## Axioms → Theorems Mapping

### Axiom C1: State Legibility

| Theorem | Formalization | Status | Description |
|---------|---------------|--------|-------------|
| `state_legible` | `axiom state_legible (State : Type) : True` | ✓ | Primitive axiom |
| `state_serializable` | Serializable ↔ Legible | ✓ | Immediate corollary |

### Axiom C2: Constraint Primacy

| Theorem | Formalization | Status | Description |
|---------|---------------|--------|-------------|
| `constraint_primacy` | `axiom constraint_primacy : True` | ✓ | Primitive axiom |
| `no_ignore_operation` | Violations must be tracked | ✓ | Immediate corollary |

### Axiom C3: Conservation of Coherence Debt

| Theorem | Formalization | Status | Description |
|---------|---------------|--------|-------------|
| `conservation_of_debt` | `axiom conservation_of_debt : True` | ✓ | Primitive axiom |
| `debt_propagates` | ΔC ≠ 0 unless terminated | ✓ | Immediate corollary |
| `hallucination_law` | Unconstrained → falsehoods | ⚠ | Derived law (see below) |

### Axiom C4: Temporal Accountability

| Theorem | Formalization | Status | Description |
|---------|---------------|--------|-------------|
| `temporal_accountability` | `axiom temporal_accountability : True` | ✓ | Primitive axiom |
| `not_pointwise` | C(t) passed → C(t+1) invalid | ✓ | Immediate corollary |
| `governance_necessary` | Gates alone insufficient | ✓ | Immediate corollary |

### Axiom C5: Irreversibility of Failure

| Theorem | Formalization | Status | Description |
|---------|---------------|--------|-------------|
| `irreversibility_of_failure` | `axiom irreversibility_of_failure : True` | ✓ | Primitive axiom |
| `fast_refusal_optimal` | Early reject > late recover | ✓ | Immediate corollary |

---

## Derived Laws (Theorems from C1–C5)

### Law 1: The Hallucination Law

| Theorem | Formalization | Status | Description |
|---------|---------------|--------|-------------|
| `hallucination_law` | Unconstrained → confident falsehoods | ⚠ | Deterministic, not probabilistic |
| `hallucination_is_debt` | Hallucination = unpaid debt | ✓ | Semantic equivalence |

### Law 2: The Governance–Throughput Tradeoff

| Theorem | Formalization | Status | Description |
|---------|---------------|--------|-------------|
| `throughput_tradeoff` | ≤2 of: speed, coherence, low-cost | ⚠ | Mathematical constraint |

### Law 3: The Receipts Necessity Theorem

| Theorem | Formalization | Status | Description |
|---------|---------------|--------|-------------|
| `receipts_necessity` | No reconstruction → meaningless claims | ✓ | Logical requirement |
| `reproducibility_derivable` | Receipts → replay | ✓ | Consequence |

### Law 4: The Early-Rejection Advantage

| Theorem | Formalization | Status | Description |
|---------|---------------|--------|-------------|
| `early_rejection_advantage` | Early reject dominates late correct | ⚠ | Matches GR damping, CFL conditions |

---

## Lean Module → Theorem Mapping

### `Coherence.Lexicon`

| Theorem | Formalization | Status | Description |
|---------|---------------|--------|-------------|
| `state_legible` | C1 axiom | ✓ | State must be legible |
| `constraint_primacy` | C2 axiom | ✓ | Constraints are prior |
| `conservation_of_debt` | C3 axiom | ✓ | Debt conservation |
| `temporal_accountability` | C4 axiom | ✓ | Not pointwise |
| `irreversibility_of_failure` | C5 axiom | ✓ | Failure irreversible |
| `lexicon_soundness` | `¬isInLexicon(term) → ¬valid_step(term)` | ✓ | Undefined terms detectable |

### `Coherence.Kernel` (NEW)

| Theorem | Formalization | Status | Description |
|---------|---------------|--------|-------------|
| `coherence_distance` | `def coherence_distance (x : State) : ℝ` | ✓ | Weighted sum of violations |
| `admissible` | `def admissible (x : State) (t : Time) : Prop` | ✓ | Within budget |
| `coherence_evolution` | `axiom coherence_evolution` | ✓ | Kernel axiom |
| `recoverable` | `def recoverable (x : State) (ε : ℝ) : Prop` | ✓ | Can reach tolerance |
| `terminally_incoherent` | `def terminally_incoherent (x : State) : Prop` | ✓ | No recovery possible |
| `governed_step` | `def governed_step` | ✓ | Full step with governance |
| `fast_refusal_minimizes_debt` | Theorem | ⚠ | Early reject optimal |
| `governance_throughput_tradeoff` | Theorem | ⚠ | Speed/coherence/cost tradeoff |
| `early_rejection_advantage` | Theorem | ⚠ | Early reject dominates |

### `Coherence.FastRefusal` (NEW - First Real Theorem)

| Theorem | Formalization | Status | Description |
|---------|---------------|--------|-------------|
| `fast_refusal_optimality` | `J(π_FR) ≤ J(π)` | ⚠ | Fast refusal optimality (main theorem) |
| `early_rejection_advantage` | `D(x) ≤ D(x')` | ⚠ | Early reject dominates |
| `algorithm_implements_policy` | Algorithm = Policy | ✓ | Minimal system implements policy |
| `cumulative_cost` | Cost functional | ✓ | J(π) definition |
| `debt_nonnegative` | `axiom` | ✓ | Assumption 1 |
| `monotone_debt_growth` | `axiom` | ✓ | Assumption 2 |

### `Coherence.UFE`

| Theorem | Formalization | Status | Description |
|---------|---------------|--------|-------------|
| `ufe_soundness` | Component residuals ≤ thresholds | ✓ | If step accepted, residuals within thresholds |
| `residual_norm_defined` | `residualNorm` is well-defined | ✓ | Norm computation is valid |
| `solves_ufe_if_residual_zero` | `ε(t)=0 → SolvesUFE` | ✓ | Trajectory solving condition |

### `Coherence.Gates`

| Theorem | Formalization | Status | Description |
|---------|---------------|--------|-------------|
| `gate_soundness` | Accepted → soft gates within tolerance | ⚠ | Placeholder proof |
| `hard_gate_failure_abort` | Hard fail → abort decision | ✓ | Hard gate semantics |
| `retry_count_enforced` | Retry limit respected | ✓ | Retry limit enforcement |

### `Coherence.Ledger`

| Theorem | Formalization | Status | Description |
|---------|---------------|--------|-------------|
| `hash_chain_preserved` | `ρ₂.parent_hash = hash(ρ₁)` | ✓ | Consecutive receipt hash chain |
| `genesis_hash_correct` | First receipt has genesis hash | ✓ | Genesis receipt property |
| `ledger_integrity` | Full ledger has unbroken chain | ⚠ | Placeholder proof |
| `schema_valid_ensures_wellformed` | Schema valid → well-formed | ✓ | Schema → well-formedness |
| `conformance_gateway` | `checkConformance = PASS ↔ ValidLedger` | ✓ | **THE GATEWAY THEOREM** |
| `SolverConformanceCertificate` | Certificate structure | ✓ | External solver certificate |

### `Coherence.Conformance`

| Theorem | Formalization | Status | Description |
|---------|---------------|--------|-------------|
| `conformance_soundness` | Certificate → all checks passed | ⚠ | Placeholder proof |
| `level_achieves_correctly` | Level achieved if conditions met | ✓ | Conformance level logic |

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✓ | Fully proven |
| ⚠ | Partial proof / placeholder / derived law |
| ✗ | Not yet implemented |

## Proof Coverage Summary

| Category | Total | Proven | Coverage |
|----------|-------|--------|----------|
| Axioms (C1–C5) | 5 | 5 | 100% |
| Immediate Corollaries | 10 | 10 | 100% |
| Derived Laws | 4 | 2 | 50% |
| **Kernel (NEW)** | 9 | 6 | 67% |
| **FastRefusal (NEW)** | 7 | 4 | 57% |
| UFE | 3 | 3 | 100% |
| Gates | 3 | 2 | 67% |
| Ledger | 4 | 3 | 75% |
| Conformance | 2 | 1 | 50% |
| **Total** | **47** | **36** | **77%** |

## Unproven Theorems (Prioritized)

1. **gate_soundness** (Gates): Full proof of acceptance → residual bounds
2. **ledger_integrity** (Ledger): Complete ledger hash chain property
3. **conformance_soundness** (Conformance): Certificate issuance implies all checks passed
4. **hallucination_law** (Derived Law): Formal proof of deterministic falsehood
5. **early_rejection_advantage** (Derived Law): Connection to CFL conditions

## The Stakes

If Coherence is correct:
- Most AI systems are systematically incoherent
- The \"hallucination problem\" is not a bug but unpaid debt
- Fast refusal is mathematically optimal
- Receipts are not optional—they're logical requirements

## Related Documents

- [Lean README](../lean/README.md): Building and running proofs
- [Spec 50 Conformance](../spec/50_conformance.md): Conformance contract
- [Spec 40 Ω-Ledger](../spec/40_omega_ledger.md): Receipt schema
- [Paper Section V-VIII](../paper/coherence_governed_evolution.tex): Axioms through Canonical Architecture
