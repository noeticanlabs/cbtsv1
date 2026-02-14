---
title: "Coherence Axioms (L0)"
description: "Core axioms defining the persistence contract for coherent systems"
last_updated: "2026-02-07"
authors: ["NoeticanLabs"]
tags: ["coherence", "axioms", "persistence-contract", "foundations"]
---

# Coherence Axioms (L0)

Coherence is a **persistence contract**: a system persists as “itself” iff its

dynamics, constraints, and evidence remain mutually compatible under evolution.

## A0 — Typed State (No Ghost Variables)
State lives in a declared space:
- continuous: x ∈ X (Banach/Hilbert/manifold)
- discrete: x ∈ Σ
- hybrid: x = (x_c, x_d) ∈ X × Σ

**Rule:** any variable influencing behavior must appear in state **and** receipts.

## A1 — Closure (No Smuggling)
There exists an evolution map Φ such that:

x⁺ = Φ_Δt(x, u, ξ)

where u is bounded control input and ξ is declared forcing/noise.

**Rule:** if an effect is real, it enters through u or ξ with units + bounds.

## A2 — Measurability (Coherence Has Numbers)
There exists a residual map r: X → ℝ^m:

r(x) = [r_phys, r_cons, r_sem, r_tool, r_ops, …].

**Rule:** “coherent” is undefined unless r(x) is computed and logged.

## A3 — Debt Functional (Scalar Summary)
There exists C: X → ℝ_{≥0}:

C(x) = Σ_i w_i ||r_i(x)||² + Σ_j v_j p_j(x),

where p_j are penalties (thrash, contradiction, missing evidence).

**Rule:** acceptance is governed by C and selected components of r.

## A4 — Bounded Interventions (Rails)
There exists a bounded action set 𝒜. Each action:
- has a trigger condition,
- has bounded magnitude,
- cannot create undefined states by itself,
- emits an auditable “before/after” record.

## A5 — Evidence Emission (Receipts)
Every step attempt emits a receipt containing:
- state summary/hash,
- residuals and debt decomposition,
- gate verdicts and actions.

**Rule:** if it isn’t receipted, it didn’t happen.

## A6 — Hard Invariants (Non-Negotiable)
There exists I_hard(x) that must never be violated on accepted states
(e.g., NaN-free, positivity, determinant constraints, bounded domain constraints).

Fail ⇒ rollback/abort (not “best effort”).

## A7 — Projection Legality (Layer Discipline)
Cross-layer claims must follow declared projection rules (see manifest).
Reverse projection is forbidden unless explicitly proven/certified.
