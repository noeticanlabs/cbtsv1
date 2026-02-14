---
title: "State Spaces and Invariants"
description: "Metric, normed, and manifold formulations with hard invariant definitions"
last_updated: "2026-02-07"
authors: ["NoeticanLabs"]
tags: ["coherence", "state-spaces", "invariants", "manifolds", "metrics"]
---

# State Spaces and Invariants

This spine works with any of the following, provided you state which case you’re in.

## 1) Metric-state formulation (most general)
Assume \((X,d_X)\) is a metric space. This is enough to define:
- bounded rails (\(d_X(a(x),x)\le \delta\)),
- continuity assumptions,
- convergence statements.

## 2) Normed/Banach formulation (for analysis)
Assume \(X\) is a normed vector space (Banach if completeness needed).
Then:
- Lipschitz assumptions on residual maps are natural,
- contraction arguments become standard,
- Grönwall-type bounds apply for ODE/PDE discretizations.

## 3) Manifold formulation (geometric systems)
Assume \(X\) is a smooth manifold with a Riemannian metric \(g\) inducing distance \(d_g\).
Then:
- rails must be defined in charts or via exponential maps,
- invariants may be geometric (e.g., \(\det \gamma > 0\) for a metric tensor).

## Hard invariants
Hard invariants are predicates \(I_{\text{hard}}:X\to\{\text{true},\text{false}\}\).
They are “non-negotiable”: accepted states must satisfy them.

Common patterns:
- domain: positivity (\(\rho\ge 0\)), bounded parameters (\(\kappa\in[\kappa_{\min},\kappa_{\max}]\))
- regularity: NaN/Inf-free, bounded condition number (if declared)
- geometry: \(\det(\gamma)>0\), orientation constraints
- conservation within absolute tolerance (safety-critical applications)

## Soft invariants
Soft invariants are inequality constraints that admit repair:
\[
q_\ell(x) \le \tau_\ell.
\]
They often include:
- equation defects,
- constraint norms,
- balance drift,
- operational thrash metrics.

## Legality axiom for rails
A rail \(a\) is **hard-legal** if, whenever invoked under its trigger conditions,
it maps admissible states to admissible states:
\[
x \in \mathcal D,\ I_{\text{hard}}(x)=\text{true} \implies I_{\text{hard}}(a(x))=\text{true}.
\]
(If a rail can violate hard invariants, that is a design bug, not a theorem.)

## 4) UFE Operator Package (Canonical Decomposition)

The Universal Field Equation (UFE) decomposes evolution into three canonical operators:

\[
\dot{\Psi}(t) = L_{\mathrm{phys}}[\Psi(t)] + S_{\mathrm{geo}}[\Psi(t)] + \sum_{i \in \mathcal{I}} G_i[\Psi(t)]
\]

### 4.1 Operator Definitions

| Operator | Symbol | Purpose |
|----------|--------|---------|
| **Physics operator** | \(L_{\mathrm{phys}}\) | Core evolution law (PDE/ODE/constraint flow) |
| **Geometry operator** | \(S_{\mathrm{geo}}\) | Geometry/gauge/constraint correction |
| **Drive operators** | \(G_i\) | External inputs, control, semantic actuation |

### 4.2 UFE in Coherence Context

The UFE structure provides the foundation for coherence enforcement:

- **Analytic coherence**: \(\varepsilon(t) = \dot{\Psi}(t) - \mathrm{RHS}(t) = 0\)
- **Operational coherence**: Bounded discrete residual \(\|\varepsilon_\Delta(t)\| \le \tau_\Delta\)
- **BridgeCert**: Certified connection between discrete and analytic bounds

### 4.3 Lean Reference

See [`lean/UFEOp.lean`](lean/UFEOp.lean) for the formal Lean 4 implementation.

## 5) GR Observer Specialization

For General Relativity observers/worldlines, the UFE residual splits into two components:

\[
\varepsilon_{\mathrm{obs}}(\tau) = \big(\underbrace{\nabla_u u}_{\text{dynamical}},\; \underbrace{g(u,u)+1}_{\text{clock}}\big)
\]

- **Dynamical coherence**: Motion satisfies geodesic/forced law
- **Clock coherence**: Parameterization normalized to Lorentzian metric (\(g(u,u) = -1\))

See [`lean/GRObserver.lean`](lean/GRObserver.lean) for the formal treatment.
