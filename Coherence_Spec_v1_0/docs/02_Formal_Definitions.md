---
title: "02 Formal Definitions"
description: "Formal definitions of state, residuals, debt, gates, hysteresis, rails, and accepted steps"
last_updated: "2026-02-07"
authors: ["NoeticanLabs"]
tags: ["coherence", "definitions", "formal", "state", "residuals"]
---

# 02 Formal Definitions

## State and Levels
- **State** \(x\): whatever is updated (fields, parameters, symbolic claims, plans).
- **Levels** \(z_\ell\) (optional): multi-resolution or hierarchical views of \(x\).

## Residuals
- **Residual vector** \(\mathbf r(x)\): measurable constraint defects.
- Examples: conservation violations, inconsistency, reconstruction error, tool mismatch, contradiction.

## Coherence Debt
Define coherence debt as:

\[
\mathfrak C(x)=\sum_i w_i r_i(x)^2 + \sum_j v_j p_j(x)
\]

where \(p_j\) are explicit penalties (thrash, ungrounded steps, excessive branching).

## Gates, Hysteresis, Rails, Receipts
- **Hard gates:** must pass or the step is rejected (e.g., NaN/Inf, forbidden states).
- **Soft gates:** policy thresholds for residuals or debt.
- **Hysteresis:** separate enter/exit thresholds to prevent chatter.
- **Rails:** bounded mitigation actions (reduce step, damp, project, backtrack, fetch evidence).
- **Receipts:** immutable records of state summaries, residuals, decisions, and actions.

## Accepted Step (precise)
A step is **accepted** if:
1. All **hard gates pass**, and
2. **Soft policy holds** after any bounded rails are applied.
