---
title: "05 Coherence Transport and Balance"
description: "Discrete balance equation for coherence change with optional continuous transport form"
last_updated: "2026-02-07"
authors: ["NoeticanLabs"]
tags: ["coherence", "transport", "balance", "discrete", "continuous"]
---

# 05 Coherence Transport and Balance

## Discrete Balance (CANON)
Per step, the coherence change must be logged as:

\[
\Delta \mathfrak C = (\text{proposal cost}) - (\text{correction benefit}) + (\text{penalties})
\]

The receipt must record each term explicitly.

## Continuous Form (BRIDGE / optional)
A structural template for continuous systems:

\[
\partial_t C_{\text{loc}} + \nabla\cdot \mathbf J_C = \Sigma - D
\]

This is **BRIDGE/UNVERIFIED** unless derived and validated for a specific system.
