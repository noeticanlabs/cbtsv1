---
title: "Debt Functionals and Coercivity"
description: "Debt functional definitions, decomposition, coercivity, and Lyapunov-augmented coherence"
last_updated: "2026-02-07"
authors: ["NoeticanLabs"]
tags: ["coherence", "debt", "functionals", "coercivity", "lyapunov"]
---

# Debt Functionals and Coercivity

## Definition (Debt)
Let normalized residuals \(\tilde r_i(x)\) and penalties \(p_j(x)\ge 0\). Define:
\[
\mathfrak C(x)=\sum_i w_i\|\tilde r_i(x)\|^2 + \sum_j v_j p_j(x).
\]

## Debt decomposition
Write \(\mathfrak C(x)=\sum_i \mathfrak C_i(x)\) where \(\mathfrak C_i=w_i\|\tilde r_i\|^2\) plus penalty terms.
This makes targeted repair mathematically sensible: repair the dominant block first.

## Coercivity (what makes debt control meaningful)
A debt functional is **coercive around a target set** \(\mathcal T\subset X\) if there exist constants
\(\alpha>0,\beta\ge 0\) such that
\[
\mathfrak C(x)\ge \alpha\, d_X(x,\mathcal T)^2 - \beta.
\]
Then bounding \(\mathfrak C\) bounds distance to \(\mathcal T\).

Common target sets:
- exact solution manifold for constraints: \(\mathcal T=\{x:c(x)=0\}\)
- consistent proof state set (no semantic/tool violations)

## Augmented Lyapunov coherence
If there exists a Lyapunov function \(V(x)\ge 0\) for the physical dynamics, define:
\[
\bar{\mathfrak C}(x) := \mathfrak C(x) + \beta V(x).
\]
This binds “truth” to “stability”.

## Units sanity
Weights and scales must make each \(\mathfrak C_i\) dimensionless.
If \(r_i\) has units, divide by \(s_i\) with the same units before squaring.
