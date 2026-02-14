import Mathlib.Data.Bool
import NoeticanLabs.UFE.UFEOp

/-!
# Gates and Governance

Formalization of gate predicates, acceptance conditions, and rail actions.

This module defines the governance layer that enforces affordability constraints.
-/

namespace NoeticanLabs.Coherence

/-- Gate decision result -/
inductive GatePass where
  | pass
  | fail
  deriving Repr, DecidableEq

/-- Failure classification for gates -/
inductive FailureClass where
  | hard      -- Must pass, abort on failure
  | soft      -- Policy boundary, retry on failure
  | warning   -- Advisory, log only
  deriving Repr, DecidableEq

/-- Gate evaluation result -/
structure GateResult where
  pass : Bool
  failure_class : Option FailureClass
  reason : Option String
  residual_value : Option ℝ
  threshold : Option ℝ
  deriving Repr, DecidableEq

/-- A gate is a predicate over state and metrics -/
structure Gate (State Metrics : Type) where
  name : String
  evaluate : State → Metrics → GateResult

/-- Soft gate configuration with tolerance -/
structure SoftGateConfig where
  name : String
  residual_key : String
  threshold : ℝ
  tolerance : ℝ  -- Fractional tolerance (e.g., 0.1 = 10%)
  retry_limit : ℕ
  deriving Repr, DecidableEq

/-- Governance configuration -/
structure GovernanceConfig where
  hard_gates : List (Gate _ _)
  soft_gates : List SoftGateConfig
  deriving Repr

/-- Step decision -/
inductive StepDecision where
  | accept
  | retry
  | abort
  deriving Repr, DecidableEq

/-- Rail action for correction -/
structure RailAction where
  rail : String
  delta_norm : ℝ
  energy_change : ℝ
  deriving Repr, DecidableEq

/-- A rail is a bounded repair map -/
structure Rail (State Metrics : Type) where
  name : String
  max_delta : ℝ
  apply : State → Metrics → RailAction

/-- Hard gate evaluation -/
def evaluateHardGate {State Metrics : Type}
  (g : Gate State Metrics) (state : State) (metrics : Metrics) : GateResult :=
  g.evaluate state metrics

/-- Soft gate evaluation with tolerance -/
def evaluateSoftGate
  (config : SoftGateConfig) (residual : ℝ) : GateResult :=
  let effective_threshold := config.threshold * (1 + config.tolerance)
  {
    pass := residual ≤ effective_threshold,
    failure_class := if residual > config.threshold then some .soft else none,
    reason := if residual > config.threshold then some "Soft gate threshold exceeded" else none,
    residual_value := some residual,
    threshold := some config.threshold
  }

/-- Check if all hard gates pass -/
def allHardGatesPass {State Metrics : Type}
  (state : State) (metrics : Metrics)
  (config : GovernanceConfig) : Bool :=
  config.hard_gates.all (fun g =>
    (evaluateHardGate g state metrics).pass)

/-- Check if all soft gates pass -/
def allSoftGatesPass {State Metrics : Type}
  (metrics : Metrics)
  (config : GovernanceConfig)
  (getResidual : String → Option ℝ) : Bool :=
  config.soft_gates.all (fun cfg =>
    match getResidual cfg.residual_key with
    | some r => (evaluateSoftGate cfg r).pass
    | none => false)

/-- Evaluate governance decision -/
def evaluateGovernance {State Metrics : Type}
  (state : State) (metrics : Metrics)
  (config : GovernanceConfig)
  (getResidual : String → Option ℝ)
  (applyRail : Rail State Metrics → State → Metrics → RailAction) :
  StepDecision × List GateResult × List RailAction :=
  let hard_results := config.hard_gates.map (evaluateHardGate · state metrics)
  let soft_results := config.soft_gates.map (fun cfg =>
    match getResidual cfg.residual_key with
    | some r => evaluateSoftGate cfg r
    | none => { pass := false, failure_class := some .soft, reason := some "Missing residual", residual_value := none, threshold := none })

  let hard_passed := hard_results.all (·.pass)
  let soft_passed := soft_results.all (·.pass)

  if hard_passed then
    if soft_passed then
      (.accept, hard_results ++ soft_results, [])
    else
      (.retry, hard_results ++ soft_results, [])
  else
    (.abort, hard_results ++ soft_results, [])

/-- Acceptance condition: all hard gates pass AND all soft gates within tolerance -/
def acceptanceCondition {State Metrics : Type}
  (state : State) (metrics : Metrics)
  (config : GovernanceConfig)
  (getResidual : String → Option ℝ) : Prop :=
  allHardGatesPass state metrics config ∧
  allSoftGatesPass metrics config getResidual

/-- Gate soundness theorem: if step is accepted, all residuals within thresholds -/
theorem gate_soundness {State Metrics : Type}
  (state : State) (metrics : Metrics)
  (config : GovernanceConfig)
  (getResidual : String → Option ℝ)
  (h_accept : acceptanceCondition state metrics config getResidual) :
  ∀ (cfg : SoftGateConfig), cfg ∈ config.soft_gates →
    match getResidual cfg.residual_key with
    | some r => r ≤ cfg.threshold * (1 + cfg.tolerance)
    | none => True := by
  simp [acceptanceCondition, allSoftGatesPass] at h_accept
  sorry  -- Placeholder for full proof

end Coherence
