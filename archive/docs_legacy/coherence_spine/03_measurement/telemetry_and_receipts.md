---
title: "Telemetry and Receipts (L2→L4)"
description: "Telemetry requirements, receipt schema, and hash chaining for audit trails"
last_updated: "2026-02-07"
authors: ["NoeticanLabs"]
tags: ["coherence", "telemetry", "receipts", "audit", "hash-chain"]
---

# Telemetry and Receipts (L2–L4)

Telemetry is the sensor layer; receipts are the audit layer.

## Minimum telemetry per step
- residual blocks (phys/cons/balance/tool/ops)
- debt: C_total and block decomposition
- hard invariants and selected soft invariants
- scheduler: chosen dt and limiter reason
- enforcement: retries + rails applied
- provenance: version/seed/platform
- integrity: hash chain

## Receipt field ordering (for deterministic hashing)
1) header
2) state
3) metrics
4) gates
5) actions
6) provenance
7) integrity

## Hash chaining rule
Let canonical_json be sorted-key, whitespace-free UTF-8.

H_k = sha256(canonical_json(R_k) || H_{k−1}).

## Rolling window summaries (for dashboards / grant reports)
- accept rate
- max residual by block
- median residual
- retry rate and rollback rate
- time-in-band (green/yellow/red)
