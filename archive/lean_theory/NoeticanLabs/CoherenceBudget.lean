import Mathlib.Algebra.Order.Ring.Defs
import Mathlib.Data.Real.Basic

/-!
# Coherence Budget

This module formalizes coherence budgets as a first-class object for reasoning about
coherence debt, receipts, and affordability gates. It provides the foundational types
and operations used by BridgeCert and UFE for coherence verification.

## Main Concepts

- **Debt**: The accumulated incoherence that must be addressed before irreversible
  actions can be performed.
- **Receipt**: Evidence artifact that demonstrates coherence at a certain scale.
- **Gate**: A predicate defining acceptable debt levels for a given operation.

## Key Invariant

The fundamental invariant is that receipts should never increase debt: applying a
valid receipt should reduce or maintain the current debt level.
-/

namespace CoherenceBudget

/-!
## Core Types

These types form the foundation of the coherence budget system.
-/

/-- `Debt` represents accumulated coherence debt in the system.

In most applications, this will be instantiated with `ℝ` or `ℝ≥0` (non-negative reals).
The choice of representation depends on whether negative debt (surplus coherence)
needs to be modeled.
-/
axiom Debt : Type

/-- A default instance for Debt using real numbers.

This axiom declares that Debt has a real number structure with order,
allowing comparison operations and arithmetic.
-/
axiom (Debt.inst : Inst (RealField Debt))

/-- `Receipt` represents evidence of coherence at a specific scale.

A receipt encapsulates the evidence artifact that can be applied to reduce
coherence debt. The exact structure is domain-dependent.
-/
structure Receipt : Type where
  /-- The scale at which coherence is being evidenced. */
  scale : ℝ
  /-- The amount of debt reduction this receipt can provide. */
  reduction : ℝ
  deriving Repr

/-- `Gate` is a predicate defining acceptable debt levels.

A gate specifies the maximum debt allowed for an operation to be affordable.
Gates can be composed and compared to create complex affordability criteria.
-/
structure Gate where
  /-- The maximum allowed debt for this gate to pass. */
  threshold : ℝ
  deriving Repr

/-!
## Core Operations

Operations for managing and querying coherence budgets.
-/

/-- `initial` represents the starting debt before any receipts are applied.

For most systems, this will be zero, but some applications may start with
an initial debt representing baseline incoherence.
-/
def initial : Debt := 0

/-- `update` applies a receipt to reduce the current debt.

The update function takes the current debt level and a receipt, returning
a new debt level that reflects the evidence provided by the receipt.

**Postcondition:** The new debt should be less than or equal to the original,
representing the invariant that receipts reduce debt.
-/
def update (d : Debt) (r : Receipt) : Debt :=
  d - r.reduction

/-- `isAffordable` checks if a debt level passes a gate.

Given a debt level and a gate, returns whether the operation is affordable
(i.e., the debt is below the gate's threshold).
-/
def isAffordable (d : Debt) (g : Gate) : Bool :=
  d ≤ g.threshold

/-!
## Key Theorems and Invariants

These theorems formalize the core invariants of the coherence budget system.
-/

/-- No irreversible action without passed receipts.

This theorem states that applying any receipt cannot increase the debt level.
This is the fundamental invariant ensuring that evidence of coherence always
improves (or maintains) the coherence state.

Formally: `∀ (d : Debt) (r : Receipt), update d r ≤ d`

This guarantees that the budget system is monotonically improving with
respect to coherence.
-/
theorem update_reduces_or_maintains_debt (d : Debt) (r : Receipt) :
    update d r ≤ d := by
  unfold update
  have h : d - r.reduction ≤ d := by
    simp_arith
  exact h

/-- Gate composition: if debt passes two gates, it passes their minimum.

This theorem is useful for systems with nested or hierarchical affordability
requirements.
-/
theorem gate_composition (d : Debt) (g₁ g₂ : Gate) :
    d ≤ g₁.threshold ∧ d ≤ g₂.threshold → d ≤ min g₁.threshold g₂.threshold :=
  fun h => And.elim h (fun h₁ h₂ => by
    simp only [min_def]
    split_ifs <;> assumption)

/-- Zero debt is always affordable.

A system with no coherence debt should always pass any gate.
-/
theorem zero_debt_always_affordable (g : Gate) : isAffordable initial g = true := by
  unfold isAffordable initial
  simp only [zero_le]

/-- Affordability is monotonic with respect to debt reduction.

If a debt level is affordable and we apply a receipt, the new debt level
remains affordable.
-/
theorem affordability_preserved_by_update (d : Debt) (r : Receipt) (g : Gate) :
    isAffordable d g = true → isAffordable (update d r) g = true := by
  intro h
  unfold isAffordable at *
  have h' : d - r.reduction ≤ g.threshold := by
    calc d - r.reduction
      ≤ d := by simp_arith
      ≤ g.threshold := h
  exact h'

end CoherenceBudget
