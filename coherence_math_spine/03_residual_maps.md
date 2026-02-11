---
title: "Residual Maps"
description: "Definition and examples of residual maps for physics, constraints, balance, and semantic defects"
last_updated: "2026-02-07"
authors: ["NoeticanLabs"]
tags: ["coherence", "residual-maps", "defects", "physics", "semantics"]
---

# Residual Maps

## Definition (Residual map)
A residual map is a function \(r:X\to \mathbb R^m\) whose components quantify violations of contracts.

## Typical physics defect (discrete)
For an evolution law \(\dot x = F(x,t)\) and a one-step proposal \(x^+\),
the defect is:
\[
r_{\text{phys}}(x,x^+,t,\Delta t) := \frac{x^+-x}{\Delta t} - F(x,t).
\]
In practice one reduces this to a scalar metric such as RMS or max norm.

## Constraint residual
If constraints are \(c(x)=0\), set \(r_{\text{cons}}(x):=c(x)\).

## Balance drift residual
If an invariant is \(Q(x)\), define relative drift:
\[
r_Q := \frac{Q(x^+)-Q(x)}{\max(|Q(x)|,Q_{\text{floor}})}.
\]

## Semantic/tool residuals (symbolic systems)
- \(r_{\text{sem}}\): type errors, contradiction count, illegal layer projections.
- \(r_{\text{tool}}\): missing citations when required, staleness flags, unresolved conflicts.

## Operational residuals
- retries, rollbacks, saturation, oscillatory dt, etc.

## Regularity assumptions (what you need for theorems)
You can pick one of these regimes:

### R1 (Continuity regime)
Each monitored scalar metric \(q_\ell(x)\) is continuous on admissible states.

### R2 (Lipschitz regime)
There exist \(L_\ell\) such that:
\[
|q_\ell(x)-q_\ell(y)|\le L_\ell d_X(x,y).
\]
This supports robustness bounds (“small state changes can’t hide big gate violations”).

### R3 (Differentiable regime)
On a Banach space, \(q_\ell\) is Fréchet differentiable, enabling local linearization:
\[
q_\ell(x+\delta) \approx q_\ell(x) + Dq_\ell(x)[\delta].
\]

## 4) UFE Residual Definition

The Universal Field Equation (UFE) provides a canonical residual structure.

### 4.1 Analytic Residual

Given a trajectory \(\Psi: \mathcal{T} \to X\), the analytic residual is:

\[
\varepsilon(t) := \dot{\Psi}(t) - \Big(L_{\mathrm{phys}}[\Psi(t)] + S_{\mathrm{geo}}[\Psi(t)] + \sum_{i \in \mathcal{I}} G_i[\Psi(t)]\Big)
\]

A trajectory is **analytically coherent** if \(\forall t,\ \varepsilon(t) = 0\).

### 4.2 Discrete Residual (Runtime)

On a finite machine, we compute a discrete derivative \(D_\Delta\):

\[
\varepsilon_\Delta(t) := D_\Delta \Psi(t) - \mathrm{RHS}(t)
\]

where \(D_\Delta\) is a forward/backward/RK-stage derivative operator.

### 4.3 Operational Coherence Gate

A step passes the coherence gate if:

\[
\|\varepsilon_\Delta(t)\| \le \tau_\Delta
\]

with accompanying receipt containing:
- Timestamp \(t\)
- Step size \(\Delta\)
- Residual norm \(\|\varepsilon_\Delta\|\)
- Threshold \(\tau_\Delta\)
- Pass/fail verdict

### 4.4 Receipt Schema

See [`../Coherence_Spec_v1_0/schemas/coherence_receipt.schema.json`](../Coherence_Spec_v1_0/schemas/coherence_receipt.schema.json) for the JSON schema.
