---
title: "Receipts, Ledgers, and Certificates"
description: "Specification for receipt structure, hash chaining, run certificates, and replay requirements"
last_updated: "2026-02-07"
authors: ["NoeticanLabs"]
tags: ["coherence", "receipts", "ledgers", "certificates", "audit"]
---

# 09 Receipts, Ledgers, and Certificates

## Receipt Required Fields
A receipt must log:
- State summary
- Residual vector and scalar debt
- Debt decomposition terms
- Gate status (hard/soft)
- Actions with before/after + bounds
- Decision (accept/retry/abort)
- Parent hash
- Receipt hash

## Hash Chaining Rule
Receipts must be chained:

\[
 h_n = H(\text{receipt}_n \,\|\, h_{n-1})
\]

## Certificate Definition
A **run certificate** summarizes a run:
- Pass/fail
- Maxima of \(\mathfrak C\) and residuals
- Final hash
- Config hash

## Replay Requirements
Receipts must contain enough data to recompute decisions and verify hashes under the same inputs/config/seed.

## BridgeCert Requirements
A **BridgeCert** is required for irreversible actions. It certifies that discrete residuals imply analytic bounds.

### BridgeCert Fields
| Field | Type | Description |
|-------|------|-------------|
| `certificate_id` | string | Unique identifier |
| `error_bound_function` | string | Formula name (e.g., "forward_euler_lipschitz") |
| `error_bound_params` | object | Parameters for the error bound |
| `verification_proof` | string | Reference to Lean proof or verification artifact |

### BridgeCert Validation
1. Check that `certificate_id` references a valid, non-revoked certificate
2. Verify `error_bound_function` is approved for the scheme
3. Confirm `verification_proof` exists and is current
4. Ensure receipt residual ≤ τ_Δ implies analytic residual ≤ τ_C

## UFE-Specific Receipt Fields
For UFE-based systems, receipts should include:

| Field | Type | Description |
|-------|------|-------------|
| `ufe_residual` | object | Residual vector by component (Lphys, Sgeo, G_i) |
| `ufe_threshold` | object | Threshold by component |
| `ufe_dynamical` | number | Dynamical residual norm (GR: ∇_u u) |
| `ufe_clock` | number | Clock residual norm (GR: g(u,u) + 1) |
| `bridge_cert_id` | string | Reference to active BridgeCert |
- **UFE components**: \(\|L_{\mathrm{phys}}\|\), \(\|S_{\mathrm{geo}}\|\), \(\|G_i\|\)
- **GR observer**: \(\|\nabla_u u\|\), \(|g(u,u)+1|\)
