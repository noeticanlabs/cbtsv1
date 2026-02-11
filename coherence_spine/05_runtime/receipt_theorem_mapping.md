---
title: "Receipt to Theorem Assumptions Mapping"
description: "Binding formal math assumptions to concrete receipt fields for runtime proof checking"
last_updated: "2026-02-07"
authors: ["NoeticanLabs"]
tags: ["coherence", "receipts", "theorems", "mapping", "verification"]
---

# Receipt → Theorem Assumptions Mapping

This table binds formal assumptions in the math spine to concrete receipt fields so proofs can be checked from runtime artifacts.

| Math assumption | Meaning | Receipt evidence (required fields) |
| --- | --- | --- |
| S1 Bounded retries | Each step has a finite retry cap. | `actions.retries_used` and policy `retry_cap` recorded in provenance or config hash; verify `retries_used ≤ retry_cap`. |
| S2 Hard-legal rails | Repair actions do not violate hard invariants when invoked correctly. | `metrics.invariants` must include hard invariants before/after; `actions.rails[]` must log before/after and bounds. |
| S3 Contractive repair | Successful repair reduces debt by a contractive rule. | `metrics.debt` per attempt and `actions.rails[]` with intended target; verify `C_after ≤ γ C_before + b` for declared `(γ,b)`. |
| Accept-only-in-A | Accepted states lie inside gate policy. | `gates` with hard/soft decisions and thresholds in policy hash; verify accept implies hard pass and soft pass. |
| Deterministic replay | Same inputs/config yield identical receipts within tolerance. | `provenance.code_version`, `provenance.seed`, config hash, and `integrity` hash chain. |

## Required receipt fields
At minimum, receipts must include:
- `metrics.debt` and per-block residuals.
- `gates` with hard/soft decisions.
- `actions` with rail bounds and before/after summaries.
- `provenance` with policy/config identifiers.
- `integrity` hash chain fields.
