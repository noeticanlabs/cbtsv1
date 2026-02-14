# Coherence — Documentation Spine (v1.0.0)

**Stamp:** 2026-02-05 America/Chicago

This spine defines **Coherence** as three interlocked things:

1) **Principle (L0):** what must be true for a system to remain itself under evolution.
2) **Measurement discipline (L1–L2):** residuals + a scalar “debt” functional that quantifies drift.
3) **Implementation contract (L4–L5):** gates, rails, receipts, and tests that enforce “no silent drift”.

## What "no placeholders" means

Every file includes:

- explicit definitions and formulas,
- operational meaning (how it runs),
- failure modes (how it breaks),
- verification hooks (how you catch it).

## Design invariant: No Silent Drift

Every accepted step must satisfy gates; every rejected step must emit evidence explaining why.

## How to use

1) Read `01_principle/*` to lock definitions.
2) Implement `05_runtime/interfaces_and_schemas.md`.
3) Wire enforcement from `04_control/*`.
4) Run `06_validation/minimal_tests.py` and inspect receipts.

## Compatibility note

This spine is layer-consistent with the L0–L5 stack and projection rules defined in `00_manifest.yaml`.
