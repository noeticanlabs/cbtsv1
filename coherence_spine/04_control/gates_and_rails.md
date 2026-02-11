---
title: "Gates and Rails (L2)"
description: "Hard and soft gates, hysteresis, and bounded corrective rail actions"
last_updated: "2026-02-07"
authors: ["NoeticanLabs"]
tags: ["coherence", "gates", "rails", "control", "enforcement"]
---

# Gates and Rails (L2)

This is where Coherence becomes behavior.

## Gate taxonomy

### Hard gates (non-negotiable)
Fail ⇒ rollback/abort.
Examples:
- NaN/Inf free
- positivity (ρ ≥ 0)
- determinant constraints (det γ > 0, etc.)
- domain bounds for parameters
- memory/time hard limits

### Soft gates (repairable)
Fail ⇒ bounded rail(s) then retry.
Examples:
- phys residual RMS
- constraint residual RMS
- conservation drift
- saturation fraction

## Rails (bounded corrective actions)

### R1 — dt deflation
Trigger: phys defect too large, CFL too high, repeated soft fails.
Update: dt ← max(dt_min, α dt), α∈(0,1) (typical α=0.8).

### R2 — rollback
Trigger: any hard fail, or retry cap hit.
Update: x ← x_prev (or abort if rollback unavailable).

### R3 — projection to constraint manifold
Trigger: constraint residual dominates while phys defect is otherwise stable.
Update: x ← Π(x), where Π is a declared projection (e.g., Helmholtz).

### R4 — bounded damping/gain adjustment
Trigger: constraint residual persists; state remains stable.
Update: κ ← min(κ_max, κ + Δκ), bounded Δκ.

## Deterministic acceptance loop
Attempt → measure → hard gates → soft gates → accept or repair+retry (bounded).

## Hysteresis defaults
warn_enter = 0.75*fail_enter
warn_exit  = 0.60*fail_enter
fail_exit  = 0.80*fail_enter
