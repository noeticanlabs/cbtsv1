---
title: "Coherence Functionals (L1)"
description: "Standardized coherence measures including debt functionals and Lyapunov-augmented coherence"
last_updated: "2026-02-07"
authors: ["NoeticanLabs"]
tags: ["coherence", "functionals", "debt", "lyapunov", "measurement"]
---

# Coherence Functionals (L1)

This file standardizes reusable coherence measures across domains.

## 1) Generic quadratic debt
C(x) = Σ_i w_i ||r_i(x)||² + Σ_j v_j p_j(x).

Recommended residual blocks:
- equation defect (physics)
- constraints/invariants
- evidence/tooling
- semantic/type/projection legality
- operational thrash

## 2) Lyapunov-augmented coherence
If V(x) ≥ 0 is a stability certificate (Lyapunov function), define:

C̄(x) = C(x) + β V(x), β>0.

Use this when you want “truthfulness” (residuals) tied to “stability” (energy).

## 3) Balance-defect debt (PDE-friendly)
For energy E(t), dissipation D(t), injected power P_in(t):

r_balance(t) = dE/dt + D(t) − P_in(t)
C_balance = ∫ w(t) r_balance(t)^2 dt.

## 4) Spectral tail barrier debt (cascade monitoring)
Define tail energy E_{≥j}(t) and barrier B_j(t):

r_tail(j,t) = max(0, E_{≥j}(t) − B_j(t))
C_tail = Σ_j w_j sup_t r_tail(j,t)^2.

## 5) Symbolic/semantic debt
For typed DSL reasoning:

C_sym = w_type r_type + w_proj r_proj + w_ref r_ref.

## 6) Normalization
For portability, normalize each residual by a scale s_i:

r̃_i = r_i / s_i, then use C = Σ w_i ||r̃_i||².
