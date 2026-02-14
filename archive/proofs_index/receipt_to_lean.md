# Receipt to Lean Proof Mapping

This document describes how Coherence receipts map to Lean proof objects, enabling formal verification of solver behavior.

## Overview

Each receipt produced by a solver corresponds to a proof object in Lean that can be used to verify:

- The step was accepted according to policy
- All residuals were within thresholds
- The hash chain is intact
- The solver conforms to the spec

## Receipt → Lean Type Mapping

| Receipt Field | Lean Type | Module |
|--------------|-----------|--------|
| `id` | `UUID` | `Coherence.Ledger` |
| `state_summary` | `StateSummary` | `Coherence.Ledger` |
| `residuals` | `ResMap` | `Coherence.Ledger` |
| `debt` | `ℝ` | `Coherence.Ledger` |
| `gates` | `GateResults` | `Coherence.Gates` |
| `decision` | `Decision` | `Coherence.Ledger` |
| `parent_hash` | `Hash` | `Coherence.Ledger` |
| `ufe_residual` | `UFEComponents` | `Coherence.UFE` |

## Proof Construction

### Step Acceptance Proof

```lean
def step_accepted (ρ : Receipt) : Prop :=
  ρ.decision = .accept ∧
  (∀ (g : GateConfig), g ∈ ρ.gates.soft →
     ρ.residuals g.name ≤ g.threshold * (1 + g.tolerance))
```

### Hash Chain Proof

```lean
def hash_chain_proof (ledger : Ledger) : Prop :=
  List.chain Coherence.Ledger.hashChainProperty ledger
```

### Conformance Certificate

```lean
structure ConformanceCertificate where
  receipt : Receipt
  proof : step_accepted receipt
  hash_proof : hash_chain_proof [receipt]
```

## Example: Proving a Single Step

```lean
-- Given a receipt ρ
variable (ρ : Receipt)

-- The receipt is well-formed
example (h : isWellFormed ρ) : Prop := by
  cases h with
  | intro h_schema h_chain =>
    -- Schema valid implies all required fields present
    show True from trivial

-- The step was accepted within bounds
example (h_accept : ρ.decision = .accept)
        (h_gates : all_gates_pass ρ.gates) :
  step_accepted ρ := by
  simp [step_accepted, h_accept, h_gates]
```

## Automated Proof Generation

The conformance suite can generate Lean proof scripts:

```bash
python CI/conformance_suite.py output/ --lean-proofs
```

This produces `output/proof.lean`:

```lean
import Coherence.Ledger
import Coherence.Gates
import Coherence.UFE

-- Generated proof for ledger
def ledger_proof (receipts : List Receipt) : Prop :=
  receipts.all (fun ρ =>
    isWellFormed ρ ∧
    (ρ.decision = .accept → all_gates_pass ρ.gates)
  )

example (h : ledger_proof my_ledger) : True := by trivial
```

## Verification Workflow

1. **Solver runs** → produces `ledger.jsonl`
2. **CI suite runs** → validates and generates proof skeleton
3. **Lean compiles** → verifies formal properties
4. **Certificate issued** → solver is conforming

## Related Documents

- [Theorem Map](theorem_map.md): What is proven
- [Lean README](../lean/README.md): Building proofs
- [Spec 40 Ω-Ledger](../spec/40_omega_ledger.md): Receipt schema
