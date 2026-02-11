---
title: "Coherence Definitions (L0→L2)"
description: "Core definitions for residuals, debt, gates, rails, and receipts"
last_updated: "2026-02-07"
authors: ["NoeticanLabs"]
tags: ["coherence", "definitions", "residuals", "debt", "gates", "rails"]
---

# Coherence Definitions (L0 → L2)

## 1) Residual vector
Residual blocks (examples; you can extend but must name them):
- r_phys(x): governing-equation defect (ODE/PDE defect, discretization defect)
- r_cons(x): constraint/invariant defect (divergence, gauge, conservation drift)
- r_sem(x): semantic/type/logic defect (contradiction, illegal projection)
- r_tool(x): tool/evidence defect (missing citations when required, stale info)
- r_ops(x): operational defect (thrash, rollbacks, saturation)

Assemble: r(x) ∈ ℝ^m.

## 2) Coherence debt functional
C(x) = Σ_i w_i ||r_i(x)||² + Σ_j v_j p_j(x), with w_i, v_j ≥ 0.

Interpretation:
- C ≈ 0: system’s math, story, and evidence align.
- C large: system is drifting, unstable, or fabricating.

## 3) Gates (pass/warn/fail with hysteresis)
For a scalar metric q:
- warn_enter, warn_exit with warn_exit < warn_enter
- fail_enter, fail_exit with fail_exit < fail_enter

Hard gates: fail ⇒ rollback/abort.
Soft gates: fail ⇒ bounded repair + retry.

## 4) Rails (bounded corrective actions)
Rails are bounded transforms a: X → X:
- dt deflation
- rollback
- projection to constraint manifold
- bounded damping/gain adjustments

Every rail action must log: trigger, before, after, and intended residual reduction.

## 5) Receipt
Receipt R_k is the canonical audit record:
- state hashes/summaries
- residuals, debt breakdown
- gate verdicts and decision
- rails applied and retries
- provenance (versions, seed, platform)
- integrity (hash chain)

## 6) Optional “continuity” modeling view
You *may* model coherence as:

∂_t C_loc + ∇·J_C = S_C − D_C

but enforcement comes from gates/receipts regardless.
