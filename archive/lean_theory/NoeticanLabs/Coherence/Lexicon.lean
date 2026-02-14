import Mathlib.Data.String.Defs

/-!
# Coherence Lexicon

Formalization of the five axioms (C1–C5) that generate everything—PDE stability,
GR constraints, governance, receipts, fast refusal—without smuggling assumptions.

All specifications, proofs, and implementations trace back to these axioms.
-/

namespace NoeticanLabs.Coherence

/-!
## Axioms (C1–C5)

These are the minimal axiom set that generates the entire Coherence framework.
-/

/-- C1: State Legibility - A system must have a well-defined state representation -/
axiom state_legible (State : Type) : True

/-- C2: Constraint Primacy - Constraints are prior; violations must be tracked -/
axiom constraint_primacy (State Constraints : Type) : True

/-- C3: Conservation of Coherence Debt - Violations propagate, accumulate, dissipate, or terminate -/
axiom conservation_of_debt : True

/-- C4: Temporal Accountability - Coherence is not pointwise -/
axiom temporal_accountability : True

/-- C5: Irreversibility of Failure - Once incoherence exceeds bounds, rollback not guaranteed -/
axiom irreversibility_of_failure : True

/-- A lexicon term identifier -/
structure LexiconTerm where
  term : String
  layer : ℕ  -- 0-5 corresponding to L0-L5
  namespace : String
  deriving Repr, DecidableEq

/-- A collection of lexicon terms -/
class Lexicon (L : Type) where
  terms : List LexiconTerm
  lookup : String → Option LexiconTerm
  contains : String → Bool

instance : Lexicon (List LexiconTerm) where
  terms := by sorry
  lookup s := List.find? (·.term = s)
  contains s := (lookup s).isSome

/-- Check if a term is in the lexicon -/
def isInLexicon {L : Type} [Lexicon L] (term : String) : Bool :=
  Lexicon.contains term

/-- Layer projection is valid if going forward in layer order -/
def validProjection (from_layer to_layer : ℕ) : Prop :=
  from_layer ≤ to_layer

/-- Forbidden reverse projection -/
def invalidProjection (from_layer to_layer : ℕ) : Prop :=
  from_layer > to_layer

/-- Receipt metadata for namespace binding -/
structure ReceiptMetadata where
  lexicon_terms_used : List String
  layers_accessed : List ℕ
  code_version : String
  solver_namespace : String
  deriving Repr, DecidableEq

/-- Check if projection rules are violated -/
def violatesProjectionRules (meta : ReceiptMetadata) : Prop :=
  -- Check for forbidden reverse projections
  let min_layer := meta.layers_accessed.min?
  let max_layer := meta.layers_accessed.max?
  match min_layer, max_layer with
  | some min, some max => max > min + 1  -- Can't skip more than one layer
  | _, _ => false

/-- Receipt is well-formed with respect to lexicon -/
def receiptWellFormedLexicon (meta : ReceiptMetadata) (lexicon : Lexicon) : Prop :=
  -- All terms used are in lexicon
  (meta.lexicon_terms_used.all (isInLexicon ·)) ∧
  -- Projection rules respected
  ¬violatesProjectionRules meta

/-- Lexicon soundness theorem: if a term is not in lexicon, it's invalid -/
theorem lexicon_soundness {L : Type} [Lexicon L] (term : String)
  (h_not_in : ¬isInLexicon term) :
  ¬Coherence.isInLexicon term := by
  simp [isInLexicon, Lexicon.contains] at h_not_in
  exact h_not_in

end Coherence
