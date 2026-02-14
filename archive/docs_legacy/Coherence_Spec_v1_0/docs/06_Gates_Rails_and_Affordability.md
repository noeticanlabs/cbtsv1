---
title: "Gates, Rails, and Affordability"
description: "Operational specification for hard/soft gates, bounded rail mitigations, and acceptance semantics"
last_updated: "2026-02-07"
authors: ["NoeticanLabs"]
tags: ["coherence", "gates", "rails", "affordability", "operational"]
---

# 06 Gates, Rails, and Affordability

## Gates
- **Hard gates:** NaN/Inf, invalid invariants, forbidden states.
- **Soft gates:** residual thresholds, debt budget, policy constraints.

## Hysteresis
Use separate **enter** and **exit** thresholds to prevent chatter when near boundaries.

## Rails (bounded mitigations)
- Reduce \(\Delta t\) or step size.
- Increase damping or regularization.
- Project to feasible set (repair).
- Fetch evidence (tool call, measurement, consistency check).
- Backtrack to checkpoint.

Each rail must specify **bounds** and be recorded in receipts.

## Acceptance Semantics
**Operational time rule:** only accepted steps advance time; retries do not. Hard gate failure always rejects the step.
