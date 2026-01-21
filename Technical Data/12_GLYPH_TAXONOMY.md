# 12_GLYPH_TAXONOMY.md
# Triaxis (NPA) — Glyph Taxonomy (v1.2)

**Suite:** Triaxis (NPA) = Noetica → Praxica → Aeonica  
**Alphabets:** GHLL (Noetica) · GLLL (Hadamard/Praxica-H) · GML (Aeonica)  
**Canon maxim:** Two alphabets, one compiler seam, one ledger truth.

---

## 1) Definitions

### 1.1 Glyph (universal definition)
A **glyph** is a typed, namespaced symbol with a stable identity and a payload that is valid for exactly one role in the stack.

Every glyph belongs to:
- an **alphabet**: `GHLL | GLLL | GML`
- a **class**: `Intent | Action | Witness`
- a **namespace ID** (`gid`) that is globally unique within its alphabet

### 1.2 The three glyph classes
- **Intent glyphs (GHLL / Noetica):** declarative meaning (types, invariants, goals, policies)
- **Action glyphs (GLLL / Praxica-H):** deterministic execution primitives (opcodes, gates, budget ops)
- **Witness glyphs (GML / Aeonica):** time/memory truth records (receipts, threads, checkpoints, rollback)

---

## 2) Namespaces (canonical)

### 2.1 GHLL (Noetica) IDs
Format:
- `N:<KIND>.<PATH>`

Examples:
- `N:TYPE.field`
- `N:INV.pde.div_free`
- `N:GOAL.min_residual`
- `N:POLICY.rails_only_control`
- `N:MAP.inv.div_free.v1`

### 2.2 GLLL (Hadamard/Praxica-H) IDs
Format:
- `H<n>:r<idx>` where `n` is Hadamard order (power of 2) and `idx` is row index.

Examples:
- `H64:r53` (PROJECT)
- `H64:r56` (EMIT)

### 2.3 GML (Aeonica) IDs
Format:
- `A:<KIND>.<PATH>`

Examples:
- `A:THREAD.PHY.M.R1`
- `A:RCPT.step.accepted`
- `A:RCPT.rollback.executed`
- `A:CLOCK.policy.triaxis_v1`

---

## 3) Canonical roles & allowed behaviors

### 3.1 GHLL / Intent rules
GHLL glyphs:
- **MUST** be declarative (order-independent meaning objects)
- **MAY** include compilation hints (`lowering_hint`, `map_id`)
- **MUST NOT** directly claim execution occurred (that is GML’s job)

### 3.2 GLLL / Action rules
GLLL glyphs:
- **MUST** be deterministic primitives with fixed semantics
- **MUST** map 1-to-1 to a Hadamard row identity at a chosen order (`H64` default)
- **MUST NOT** encode high-level meaning directly (only mechanism)

### 3.3 GML / Witness rules
GML glyphs (receipts, threads):
- **MUST** be append-only truth records
- **MUST** include clock stamps and join keys back to intent and action
- **MUST** be hash-chained (ledger continuity)

---

## 4) The compiler seam (binding law)

### 4.1 Seam definition
The seam is a single function the entire suite agrees on:

\[
\mathrm{lower}:\; \mathcal{G}_{GHLL} \to (\mathcal{I}_{GLLL})^*
\]

Meaning compiles into an ordered sequence of action instructions.

### 4.2 Lowering map object (required)
A lowering map is itself a GHLL object (`N:MAP.*`) with a normalized schema:

```json
{
  "map_id": "N:MAP.inv.div_free.v1",
  "from_intent": "N:INV.pde.div_free",
  "to_ops": [
    {"op": "H64:r48", "args": {"gate": "constraint_scope"}},
    {"op": "H64:r53", "args": {"project": "leray"}},
    {"op": "H64:r50", "args": {"check": "div(v)=0"}},
    {"op": "H64:r49", "args": {}},
    {"op": "H64:r56", "args": {"emit": "A:RCPT.check.invariant"}},
    {"op": "H64:r61", "args": {"commit": "step"}}
  ],
  "requirements": {
    "must_emit_receipt": true,
    "must_tag_thread": true,
    "must_be_deterministic": true
  },
  "version": "1.2"
}
```

Rules:

* `from_intent` **MUST** be a valid GHLL glyph ID.
* Each `op` **MUST** be a valid GLLL glyph ID.
* If `must_emit_receipt` is true, the resulting execution **MUST** generate a GML receipt that includes both `intent_id` and `ops[]`.

---

## 5) Truth rule (ledger primacy)

If GHLL claims X and GLLL executes Y, **GML decides what happened**.

Formal axiom:
[
\text{Truth} \equiv \text{hash-chained Aeonica receipts (replayable)}.
]

Operational rule:

* A step is “accepted” **only** if an `A:RCPT.step.accepted` receipt exists with valid hash continuity and gate outcomes.

---

## 6) Universal glyph object schemas

### 6.1 GHLL glyph schema (normalized)

```json
{
  "id": "N:INV.pde.div_free",
  "kind": "INV",
  "domain": "NS",
  "statement": "div(v)=0",
  "scope": {"where": "velocity_field", "when": "each_step"},
  "tolerance": {"abs": "0.000000000100", "rel": "0.000000010000"},
  "lowering_hint": "use_leray_projection",
  "version": "1.2"
}
```

### 6.2 GLLL instruction schema (normalized)

```json
{
  "op": "H64:r53",
  "mode": {"predicated": false, "vectorized": false, "safe_only": true},
  "args": {"project": "leray"},
  "effects": ["reads:field:v", "writes:field:v"]
}
```

### 6.3 GML receipt schema (normalized, step receipt)

```json
{
  "run_id": "R-2026-01-19-0001",
  "step_id": 128,
  "event": "A:RCPT.step.accepted",
  "thread": "A:THREAD.PHY.M.R1",
  "t": "12.800000000",
  "dt": "0.010000000",
  "tau": "12.734500000",
  "dtau": "0.009820000",
  "intent_id": "N:INV.pde.div_free",
  "intent_hash": "2f3d4c5b22e4b6b70b1eacbbcd1a56f3f3f9e91a0b8d3abf25a4a3f9246e5f1b",
  "ops": ["H64:r48","H64:r53","H64:r50","H64:r49","H64:r56","H64:r61"],
  "gates": {"tail_barrier": {"status": "pass"}},
  "status": "accepted",
  "hash_prev": "0000000000000000000000000000000000000000000000000000000000000000",
  "hash": "dbcd3b52e6619ef5afea0679bc4bdba3d0b68a34985c8049e71d8d8719e3c34c"
}
```

---

## 7) Validation checklist (what “valid glyph” means)

A Triaxis implementation **MUST** enforce:

1. Namespace validity (`N:` vs `H` vs `A:`).
2. Role validity (GHLL cannot execute; GLLL cannot redefine meaning; GML cannot be rewritten).
3. Lowering maps exist for executable GHLL intents used at runtime.
4. Every executed step emits at least one GML step receipt with hash continuity.
5. Gate outcomes are present for every gate that was active in that step.

---

## 8) Compatibility contract

* GHLL may evolve (new meanings), but existing `N:*` IDs must be versioned, not overwritten.
* GLLL opcodes (`H64:r00..r63`) are stable identities: semantics must not change across v1.x.
* GML receipt fields may extend, but existing fields and hash rules must remain stable across v1.x.