---
title: "Gated Dynamics as a Hybrid System"
description: "Modeling coherence enforcement as a hybrid dynamical system with gates and bounded rails"
last_updated: "2026-02-07"
authors: ["NoeticanLabs"]
tags: ["coherence", "gated-dynamics", "hybrid-systems", "control"]
---

# Gated Dynamics as a Hybrid System

We model coherence enforcement as a **hybrid dynamical system**:
- a proposed evolution map (physics / reasoning step),
- gates that decide accept/reject,
- bounded rails that repair and retry.

## 1) Proposed step map
Let a proposal operator be:
\[
x^+ = \Phi_{\Delta t}(x; \eta)
\]
where \(\eta\) collects declared inputs (controls/noise/tool calls).

## 2) Gate policy
A gate policy is:
- hard predicates \(I_{\text{hard}}(x)\),
- soft metrics \(q_\ell(x)\) with thresholds and hysteresis.

Define the acceptance set:
\[
\mathcal A := \{x: I_{\text{hard}}(x)\ \wedge\ q_\ell(x)\le \tau_\ell \ \forall \ell\}.
\]

## 3) Rails
Rails are bounded maps \(a\in\mathcal R\) applied only under triggers.
A retry loop yields a sequence:
\[
x^{(0)}=x,\quad x^{(k+1)} = a_k(x^{(k)}),\quad x^{(k)}\notin \mathcal A,
\]
until either \(x^{(k)}\in \mathcal A\) (accept) or retry cap is hit (reject/rollback).

## 4) Contractive repair hypothesis (core)
A repair action sequence is **contractive in debt** if for some \(\gamma\in(0,1)\), \(b\ge 0\):
\[
\mathfrak C(x^{(k+1)}) \le \gamma\,\mathfrak C(x^{(k)}) + b.
\]
This is the exact condition needed for global boundedness theorems.

## 5) Receipt as mathematical object
Treat each attempt as producing a labeled transition:
\[
(x,\text{context}) \xrightarrow{\text{metrics, gates, rails}} (x',\text{context}')
\]
A receipt is the recorded label. In proof terms, the receipt log is a *certificate trace*.

## 6) BridgeCert Pattern (Coherence License)

A **BridgeCert** is the only place in the coherence architecture where numerical analysis is permitted. It certifies that discrete residuals imply analytic bounds.

### 6.1 BridgeCert Definition

\[
\boxed{
\|\varepsilon_\Delta(t)\| \le \tau_\Delta \;\Longrightarrow\; \|\varepsilon(t)\| \le \mathrm{errorBound}(\tau_\Delta, \Delta)
}
\]

### 6.2 Governance Rule

No irreversible actions are permitted unless:
1. Receipts show \(\|\varepsilon_\Delta\| \le \tau_\Delta\)
2. A valid BridgeCert exists for the scheme

```lean
def closesAt {Ψ : Type u}
  [NormedAddCommGroup Ψ] [NormedSpace ℝ Ψ]
  (op : UFEOp Ψ) [Fintype op.ι] [BridgeCert op]
  (ψ : Traj Ψ) (t Δ τΔ : ℝ) : Prop :=
  ‖residualΔ op ψ t Δ‖ ≤ τΔ
```

### 6.3 Lean Reference

See [`lean/BridgeCert.lean`](lean/BridgeCert.lean) for the formal typeclass.
