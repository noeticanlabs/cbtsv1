---
title: "Interfaces and Schemas (L4)"
description: "Language-agnostic interfaces and JSON schemas for state, model, gates, rails, and receipts"
last_updated: "2026-02-07"
authors: ["NoeticanLabs"]
tags: ["coherence", "interfaces", "schemas", "api", "runtime"]
---

# Interfaces and Schemas (L4)

## Minimal interfaces (language-agnostic)

### State
- serialize() -> bytes (canonical)
- summary() -> dict (bounded size)
- check_invariants() -> dict[str,bool]
- clone() -> State

### Model
- rhs(state,t)
- step(state,t,dt) -> proposed_state
- residual(state, proposed_state, t, dt) -> metrics dict

### GatePolicy
- evaluate(metrics) -> verdicts + decision

### Rail
- apply(state, metrics, context) -> (state, context)

### ReceiptEmitter
- emit(receipt_dict) -> (receipt_json, hash)

## Canonical Receipt Schema (v2.0.0) - CONSOLIDATED

**STATUS**: Unified canonical schema consolidating:
- `schemas/omega_ledger.schema.json` (v1.0 - flat structure, `hash`/`prev_hash`)
- `coherence_spine/05_runtime/interfaces_and_schemas.md` (hierarchical, `prev_hash`/`this_hash`)
- CI validators (flat, `receipt_hash`/`parent_hash`)

**CANONICAL FIELD NAMING**:
- `receipt_hash` (was `hash` or `this_hash`)
- `parent_hash` (was `prev_hash`)

See `schemas/omega_ledger.schema.json` for the complete canonical schema definition.

### Receipt Structure (Flat with Nested Objects)

```json
{
  "id": "step-0042",
  "version": "2.0.0",
  "timestamp": "2025-01-01T00:00:00Z",
  "run_id": "run-abc123",
  "step_id": 42,
  "step_size": 0.05,
  "receipt_hash": "sha256(canonical_json(receipt_without_hashes) || parent_hash)",
  "parent_hash": "sha256(...) or null for genesis",
  "state_summary": {
    "hash_before": "state-hash-t0",
    "hash_after": "state-hash-t1",
    "summary": {}
  },
  "residuals": {
    "cons": 0.9,
    "rec": 0.7,
    "tool": 0.2
  },
  "debt": 1.42,
  "debt_decomposition": {
    "cons": 0.81,
    "rec": 0.49,
    "tool": 0.04,
    "thrash": 0.08
  },
  "gates": {
    "hard": { ... },
    "soft": { ... },
    "decision": "accept|retry|abort"
  },
  "actions": [ ... ],
  "decision": "accept|retry|abort",
  "ufe_residual": {
    "Lphys": 0.001,
    "Sgeo": 0.002,
    "G_total": 0.003
  },
  "lexicon_terms_used": ["term1", "term2"],
  "namespaces": ["namespace1"],
  "layer": ["L0", "L1", "L2"],
  "artifacts": { ... },
  "code_version": "1.0.0",
  "seed": 12345,
  "notes": ["info message"]
}
```

### Deterministic Hashing Rule (CANONICAL)

For hash chain integrity verification:

```
receipt_hash = sha256(canonical_json(receipt_without_hashes) || parent_hash_bytes)

where:
  - receipt_without_hashes = receipt with receipt_hash and parent_hash fields removed
  - canonical_json = JSON with sorted keys, UTF-8 encoding, separators ',' and ':'
  - || = byte concatenation
  - parent_hash_bytes = parent_hash encoded as UTF-8 bytes (or all zeros for genesis)
```

### Hash Chain Invariants

1. **Genesis Block** (receipt index 0):
   - `parent_hash` must be `null`
   - `receipt_hash` = sha256(canonical_json(receipt) || genesis_parent_hash_bytes)

2. **Non-Genesis Blocks** (receipt index > 0):
   - `parent_hash` must equal `receipt_hash` of previous receipt
   - `receipt_hash` must match sha256(canonical_json(receipt) || parent_hash_bytes)

3. **Chain Continuity**:
   - For all receipts i > 0: `receipts[i].parent_hash == hash(receipts[i-1])`
   - Tampering with any receipt invalidates all subsequent receipts

### Lexicon and Layer Binding

Every receipt must include:
- `lexicon_terms_used[]` - lexicon terms referenced (for projection legality checking)
- `layer[]` - computation layers involved (for layer hierarchy verification)

These fields enable detection of illegal layer jumps and undefined term usage.

### Migration from Previous Schemas

**Old Field Names** → **Canonical Names**:
- `hash` → `receipt_hash`
- `prev_hash` → `parent_hash`
- `this_hash` → `receipt_hash`

**Breaking Changes**:
- Schema v1.0 receipts using old field names will fail validation against v2.0.0 schema
- Receipts must be regenerated or migrated to use canonical naming
- Hash values may change if canonical_json rule differs (version 1.0 vs 2.0.0)

See `SCHEMA_CONSOLIDATION_ANALYSIS.md` for detailed migration guidance.
