---
title: "Certificates"
description: "Inequality certificates, SOS, interval bounds, small-gain, trace certificates, and BridgeCert patterns"
last_updated: "2026-02-10"
authors: ["NoeticanLabs"]
tags: ["coherence", "certificates", "proof", "sos", "verification", "bridgecert"]
---

# Certificates (What Counts as Proof in a Governed System)

In coherence engineering, a "certificate" is a compact artifact that can be checked mechanically.

## 1) Inequality certificates
### Sum-of-squares (SOS)
To certify a polynomial inequality \(p(x)\ge 0\) on a region, provide an SOS decomposition:
\[
p(x)=\sum_i s_i(x)^2 + \sum_k \lambda_k(x) g_k(x),
\]
where \(g_k(x)\ge 0\) describe the region. Verification reduces to algebra.

### Interval arithmetic bounds
To certify numerical claims (e.g., max residual ≤ \(\tau\)), provide interval bounds:
- exact input bounds,
- interval evaluation method,
- final interval width.

## 2) Discrete invariants certificates
For a discrete scheme, certify:
- invariants preserved to tolerance,
- monotone energy decay,
- contractive repair inequality parameters \((\gamma,b)\).

## 3) Small-gain certificates
Provide \(\alpha,\beta,e_A,e_B\) and verify \(\alpha\beta<1\).
Then the bound is checkable with a few arithmetic operations.

## 4) Trace (receipt) certificates
A receipt log is a *witness trace*:
- shows every step's residuals,
- shows gate verdicts,
- shows bounded rails,
- forms a hash-chained evidence stream.

This is not "math by faith"; it is math plus a verifiable execution trace.

## 5) BridgeCert (The Coherence License)

A **BridgeCert** is the only place in the coherence architecture where numerical analysis is permitted. It certifies that discrete residuals imply analytic bounds, providing the "licensed truth from evidence."

### 5.1 BridgeCert Typeclass

```lean
class BridgeCert {Ψ : Type u}
  [NormedAddCommGroup Ψ] [NormedSpace ℝ Ψ]
  (op : UFEOp Ψ) [Fintype op.ι] where

  errorBound : ℝ → ℝ → ℝ   -- τΔ → Δ → τC

  bridge :
    ∀ (ψ : Traj Ψ) (t Δ τΔ : ℝ),
      ‖residualΔ op ψ t Δ‖ ≤ τΔ →
      ‖residual  op ψ t‖   ≤ errorBound τΔ Δ
```

### 5.2 Bridge Properties

| Property | Requirement |
|----------|-------------|
| **Isolation** | Only place numerical analysis is allowed |
| **Auditable** | Typeclass with checkable axioms |
| **Composable** | Can be stacked for multi-scale systems |
| **Governable** | Enables "no irreversible actions without cert" policy |

### 5.3 Example: Forward Euler Bridge

For forward Euler with step Δ:

errorBound(τ_Δ, Δ) = τ_Δ + O(Δ)

The exact constant depends on Lipschitz bounds of the RHS.

### 5.4 Lean Reference

See [`lean/BridgeCert.lean`](lean/BridgeCert.lean) for the formal implementation.

## 6) Proper Time Bridge (GR Specialization)

For GR observers, proper time τ relates to an arbitrary parameter λ via:

\[
\frac{d\tau}{d\lambda} = \sqrt{-g(u(\lambda), u(\lambda))}
\]

where u = dγ/dλ is the tangent vector.

### 6.1 Bridge Facts Required

1. **FTC**: Derivative of τ(λ) equals α(λ) = √(-g(u,u))
2. **Inverse function**: Derivative of λ(τ) = τ⁻¹ equals 1/(α ∘ λ)
3. **Monotonicity**: α > 0 ensures τ is strictly increasing

### 6.2 Lean Reference

See [`lean/GRObserver.lean`](lean/GRObserver.lean) for the formal proper time construction.
