---
title: "Scheduler and Multi-Clock Time (L2)"
description: "Multi-clock time stepping with debt-aware adaptation and coherence time dilation"
last_updated: "2026-02-07"
authors: ["NoeticanLabs"]
tags: ["coherence", "scheduler", "multiclock", "time-stepping", "adaptation"]
---

# Scheduler and Multi-Clock Time (L2)

## Multi-clock rule
Compute candidate step sizes:

dt_adv, dt_diff, dt_force, dt_control, dt_user

Then:

dt = min(dt_adv, dt_diff, dt_force, dt_control, dt_user).

## Typical examples
Advection-like: dt_adv ≤ CFL * dx / max(|u|)
Diffusion-like: dt_diff ≤ CFL * dx² / ν

## Debt-aware adaptation
In addition to standard stability limits, enforce coherence debt:

dt_next = clip(dt * exp(−k*(C/C_target − 1)), dt_min, dt_max), k>0.

## Interpreting “coherence time dilation”
This is not physical dilation. It is **compute budget reallocation**:
when coherence is fragile, you spend more wall-clock per simulated time.

## Receipt requirement
Every step must record:
- dt and limiter reason (at minimum)
- rails that changed dt (if any)
