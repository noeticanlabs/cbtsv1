---
title: "Failure Modes and Recovery"
description: "Taxonomy of failure modes, symptom mapping, and recovery policies for coherence systems"
last_updated: "2026-02-07"
authors: ["NoeticanLabs"]
tags: ["coherence", "failure-modes", "recovery", "operational"]
---

# 11 Failure Modes and Recovery

## Failure Mode Taxonomy
| Failure Mode | Symptoms | Recovery Policy |
| --- | --- | --- |
| Too-strict gates | \(\Delta t\) collapse, repeated retries | Widen hysteresis, staged rails |
| Too-loose gates | Drift, coherence leak | Tighten budgets, add residuals |
| Missing normalization | One residual dominates | Normalize to unit budget boundary |
| Thrash loops | Oscillation, excessive retries | Add thrash penalty, cap branching, backtrack |

## Symptom-to-Residual Mapping
- Drift → residuals grow slowly but continuously.
- Thrash → oscillatory residuals around thresholds.
- Collapse → frequent hard-gate failures.

## Recovery Policy (minimum)
- Apply bounded rails in priority order.
- Log all actions with bounds.
- Abort if retries exceed cap or \(\Delta t\) falls below floor.
