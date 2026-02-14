---
title: "04 Coherence Functionals"
description: "Required diagnostic vector and scalar debt outputs per step with canonical decomposition"
last_updated: "2026-02-07"
authors: ["NoeticanLabs"]
tags: ["coherence", "functionals", "debt", "diagnostics", "normalization"]
---

# 04 Coherence Functionals

## Required Outputs per Step
Each step must emit a diagnostic vector and scalar debt:

\[
\mathbf c = (r_1,\dots,r_m,\mathfrak C)
\]

## Canonical Decomposition (engineering default)
A recommended decomposition:

\[
\mathfrak C = w_{\text{cons}}\varepsilon^2 + w_{\text{rec}}r_{\text{rec}}^2 + w_{\text{tool}}r_{\text{tool}}^2 + w_{\text{thrash}}p_{\text{thrash}}
\]

Where:
- \(\varepsilon\): conservation defect
- \(r_{\text{rec}}\): reconstruction error
- \(r_{\text{tool}}\): tool mismatch / evidence defect
- \(p_{\text{thrash}}\): penalty for oscillation or excessive branching

## Normalization Rule
Each residual must be scaled so **1.0 equals the budget boundary**. This prevents any single residual from dominating due to units or scale.
