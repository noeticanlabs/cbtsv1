---
title: "Residuals and Metrics (L1→L2)"
description: "Standardized residual blocks for physics, constraints, conservation, evidence, and operations"
last_updated: "2026-02-07"
authors: ["NoeticanLabs"]
tags: ["coherence", "residuals", "metrics", "measurement", "physics"]
---

# Residuals and Metrics (L1–L2)

This standardizes what gets measured so enforcement is portable.

## 1) Physics residual (equation defect)
For ẋ = F(x,t), a one-step defect:

r_phys,k = (x_{k+1} − x_k)/dt_k − F(x_k,t_k).

Norm choices:
- RMS: ||r||₂/√N (good default)
- Max: ||r||_∞ (safety-critical)

## 2) Constraint residual
If constraints are c(x)=0, define r_cons=c(x).
Examples: ∇·u, gauge constraints, Hamiltonian/momentum constraints.

## 3) Conservation drift residual
For an invariant Q(x):

r_Q = (Q_{k+1} − Q_k)/max(|Q_k|, Q_floor).

## 4) Evidence/tool residuals (reasoning/documentation)
- r_ref: count of required citations missing
- r_stale: count of claims not verified when freshness is required
- r_conflict: count of unresolved contradictions

## 5) Operational residuals
- r_retry: retries used
- r_rollback: rollbacks in window
- r_sat: controller saturation fraction

## 6) Banding pattern
Green: q ≤ warn_exit
Yellow: warn_enter < q ≤ fail_enter
Red: q > fail_enter
Use hysteresis exits to prevent chatter.
