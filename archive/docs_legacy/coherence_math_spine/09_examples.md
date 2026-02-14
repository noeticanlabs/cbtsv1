---
title: "Examples"
description: "Complete minimal examples: ODE with gated stepping, constraint projection, semantic coherence"
last_updated: "2026-02-07"
authors: ["NoeticanLabs"]
tags: ["coherence", "examples", "tutorial", "ode", "constraints"]
---

# Examples (Complete, Minimal)

## Example 1 — Stable ODE with gated stepping
System: \(\dot x = -\lambda x\), \(\lambda>0\).
Explicit Euler proposal:
\[
x^+ = x - \lambda \Delta t\, x.
\]
Defect residual:
\[
r_{\text{phys}} = \frac{x^+-x}{\Delta t} + \lambda x.
\]
Debt (single-block):
\[
\mathfrak C(x) = w\, (r_{\text{phys}}/s)^2.
\]
Soft gate: \(|r_{\text{phys}}|\le \tau\).
Rail: if fail, \(\Delta t \leftarrow \alpha \Delta t\) with \(\alpha\in(0,1)\).
Then for sufficiently small \(\Delta t\), the gate will pass and debt stays bounded.

## Example 2 — Constraint projection (incompressibility template)
State \(x=(u,p)\). Constraint residual \(r_{\text{cons}}=\nabla\cdot u\).
Rail: \(u\leftarrow \Pi(u)\) (Helmholtz projection).
Debt includes both physics defect and constraint norm. Gate selects projection when constraint dominates.

## Example 3 — Semantic/tool coherence
Let a reasoning state include a set of claims with required citations.
Residual blocks:
- \(r_{\text{ref}} = \#(\text{missing citations})\),
- \(r_{\text{conflict}} = \#(\text{unresolved contradictions})\).
Debt:
\[
\mathfrak C = w_{\text{ref}} r_{\text{ref}} + w_{\text{conflict}} r_{\text{conflict}}.
\]
Gate policy: reject if \(r_{\text{ref}}>0\) in “accuracy mode”.
Rail: invoke verification tool; retry after evidence is added.
