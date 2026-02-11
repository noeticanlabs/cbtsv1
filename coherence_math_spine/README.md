# Coherence — Mathematical Spine (v1.0.0)

**Stamp:** 2026-02-05 America/Chicago

This is the **math core** behind the coherence doc spine:

- a precise residual/debt formalism,
- a hybrid “gated dynamics” model (accept/reject + bounded repair),
- theorems guaranteeing *bounded drift* and *audit-friendly stability*,
- and multiscale barrier templates (dyadic tails, slab induction).

## What this is (and isn’t)

- It **is** a reusable mathematical kernel you can plug into PDE solvers, symbolic DSLs, or mixed systems.
- It **is not** a claim that *Nature* obeys your symbols. It’s a discipline: if you enforce the hypotheses, the conclusions follow.

## Layer intent

- L1: definitions of C(x), r(x), invariants, norms
- L2: gate/rail policies as a hybrid control system
- L5: theorems + certificates

## Minimal contract

You need:

1) a state space (metric/Banach/manifold),
2) residual maps r_i(x) with scales,
3) a debt functional C(x),
4) gate sets (hard/soft) with hysteresis,
5) bounded repair actions (rails),
6) receipts (for proofs-by-audit).

## Optional Lean check

If Lean is installed, you can run:

```bash
./scripts/check_lean_smallgain.sh
```

This verifies the small-gain lemma in `coherence_math_spine/lean/SmallGain.lean`.
