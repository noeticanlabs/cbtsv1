## Two different "glyph systems"

### 1) **Noetican glyphs (HLL / Noetica)**

* **What they are:** *semantic constructs* (Intent glyphs)
* **Job:** express meaning, invariants, goals, types, domain structure
* **Identity:** "meaning-first" IDs (lexicon entries, compositional forms)

Think: *symbols in a high-level language*.

### 2) **Hadamard glyphs (LLL / Praxica-H)**

* **What they are:** *execution codewords* (Action glyph opcodes + gate tags)
* **Job:** dispatch, rails/gates, low-level instruction identity, error-tolerant transmission
* **Identity:** "mechanism-first" IDs (Hadamard row codes)

Think: *opcodes / VM bytecode identifiers*.

So yes: **Hadamard-glyph LLL is separate from Noetican-glyph HLL**.

---

## The binding seam (the thing that keeps the universe from tearing)

You need a **single, explicit mapping layer**:

### **N→P Lowering Map**

A Noetican glyph doesn't "equal" a Hadamard glyph.
It **compiles into a Praxica opcode sequence** (each opcode has a Hadamard ID).

Formally:

* **Noetica glyph ID:** `N:<concept>`
* **Praxica opcode ID:** `P:H32:rXX` (Hadamard codeword row)

Mapping:
\[
\mathrm{lower}:; N_\text{glyph} ;\longrightarrow; (P_\text{op},;args)^*
\]

Example (conceptual):

* `N:INV(div_free)`
  lowers to:

  * `P:H32:r24 GATE_B (constraint_scope)`
  * `P:H32:r26 PROJECT (leray)`
  * `P:H32:r30 CHECK (div(v)=0)`
  * `P:H32:r25 GATE_E`

---

## Aeonica's role: witness the binding

Aeonica receipts should carry **both IDs** so audits can join meaning ↔ execution:

```json
{
  "intent_id": "N:INV(div_free)",
  "lowered_ops": ["P:H32:r24","P:H32:r26","P:H32:r30","P:H32:r25"],
  "decode_margin": 22,
  "gate_result": "pass",
  "hash_prev": "…",
  "hash": "…"
}
```

This is where your system becomes *forensic*: you can prove which meaning triggered which mechanism.

---

## Canon rules (so it stays coherent)

1. **Noetica glyphs are not constrained by Hadamard size.** (meaning can grow)
2. **Praxica opcodes must have fixed Hadamard IDs.** (execution must be deterministic)
3. **Aeonica receipts record the linkage.** (evidence is non-negotiable)
4. Optional but powerful: assign Noetica glyphs a **stable semantic hash** (not Hadamard) for versioning, while Hadamard remains the opcode identity.

---

## Where this goes in the doc spine

Add a dedicated document:

**14_GLYPH_BINDING_NOETICA_TO_PRAXICA_H.md**

* namespaces: `N:` vs `P:H..:` vs `A:receipt`
* lowering rules + mapping tables
* receipt fields that "witness" the mapping

That's the clean separation: **two alphabets, one compiler seam, one ledger truth.**