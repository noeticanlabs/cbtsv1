import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Data.Real.Basic

open NoeticanLabs.UFE

namespace NoeticanLabs.UFE.Discrete

/-!
# Discrete Runtime and Receipts

This module defines the discrete residual (runtime) and receipt structures
for operational coherence enforcement.
-/

-- Discrete derivative (forward difference) for runtime
noncomputable def fdiff {Ψ : Type u}
  [NormedAddCommGroup Ψ] [NormedSpace ℝ Ψ]
  (ψ : Traj Ψ) (t Δ : ℝ) : Ψ :=
  (ψ (t + Δ) - ψ t) / Δ

-- Discrete residual: ε_Δ(t) = D_Δ Ψ(t) - RHS(t)
noncomputable def residualΔ {Ψ : Type u}
  [NormedAddCommGroup Ψ] [NormedSpace ℝ Ψ]
  (op : UFEOp Ψ) [Fintype op.ι] (ψ : Traj Ψ) (t Δ : ℝ) : Ψ :=
  fdiff ψ t Δ - rhs op ψ t

-- Norm of discrete residual
noncomputable def residualΔNorm {Ψ : Type u}
  [NormedAddCommGroup Ψ] [NormedSpace ℝ Ψ]
  (op : UFEOp Ψ) [Fintype op.ι] (ψ : Traj Ψ) (t Δ : ℝ) : ℝ :=
  ‖residualΔ op ψ t Δ‖

/-!
## Receipt Structure

A receipt records the evidence of a coherence check.
-/

structure ReceiptΔ where
  t     : ℝ          -- timestamp
  Δ     : ℝ          -- step size
  eps   : ℝ          -- residual norm
  τ     : ℝ          -- threshold
  passed : Prop      -- pass/fail verdict

-- Emit a receipt for a coherence check
noncomputable def emitReceiptΔ {Ψ : Type u}
  [NormedAddCommGroup Ψ] [NormedSpace ℝ Ψ]
  (op : UFEOp Ψ) [Fintype op.ι]
  (ψ : Traj Ψ) (t Δ τ : ℝ) : ReceiptΔ :=
  let ε := residualΔ op ψ t Δ
  let e : ℝ := ‖ε‖
  { t := t, Δ := Δ, eps := e, τ := τ, passed := e ≤ τ }

-- Coherence gate: returns true if residual is below threshold
noncomputable def coherenceGate {Ψ : Type u}
  [NormedAddCommGroup Ψ] [NormedSpace ℝ Ψ]
  (op : UFEOp Ψ) [Fintype op.ι]
  (ψ : Traj Ψ) (t Δ τ : ℝ) : Prop :=
  residualΔNorm op ψ t Δ ≤ τ

end NoeticanLabs.UFE.Discrete
