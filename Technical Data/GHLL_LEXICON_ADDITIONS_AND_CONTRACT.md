# GHLL_LEXICON_ADDITIONS_AND_CONTRACT.md
# Triaxis (NPA) — GHLL Lexicon Additions & Contract (v1.2)

**Alphabet:** GHLL (Noetica / Meaning / Intent)  
**Role:** Declare meaning, constraints, goals, and policies that compile into Praxica actions and must be witnessed by Aeonica receipts.  
**Canon maxim:** Two alphabets, one compiler seam, one ledger truth.

This document is **normative** for:
1) how new GHLL glyphs are introduced (lexicon additions),
2) what every GHLL glyph **must** specify (contract),
3) versioning + stability rules for meaning IDs,
4) how GHLL binds to the seam (`N:MAP.*`) and to Aeonica witnessing (`intent_hash`).

---

## 1) What “GHLL Lexicon” means (and what it does not)

### 1.1 GHLL lexicon = meaning registry
The **GHLL lexicon** is the authoritative registry of declarative meaning objects:

\[
\texttt{intent\_id} \;\;\longleftrightarrow\;\; \texttt{(semantics, type, constraints, proof obligations, lowering hooks)}.
\]

GHLL glyphs define **what must be true**, not what happened.

### 1.2 What GHLL is NOT
- Not an execution language (that’s Praxica / GLLL).
- Not a memory ledger (that’s Aeonica / GML).
- Not a bag of informal labels. Every GHLL entry has a **typed, hashable** definition.

---

## 2) GHLL namespace and stability rules

### 2.1 Canonical GHLL ID format
GHLL IDs are globally unique strings:

- `N:<KIND>.<PATH>`

Examples:
- `N:TYPE.field`
- `N:INV.pde.div_free`
- `N:GOAL.min_residual`
- `N:POLICY.deterministic_replay`

### 2.2 Kinds (closed set for v1.x)
GHLL kinds are a fixed taxonomy in v1.x:

- `TYPE`   — type declarations and shape constraints  
- `INV`    — invariants (“must hold”)  
- `GOAL`   — objective targets (“prefer”)  
- `POLICY` — governance constraints (rails, determinism, audit cadence)  
- `DOMAIN` — semantic domain tags  
- `SPEC`   — bundled specs (named sets of inv/goals/policies/types)  
- `MAP`    — lowering maps (compiler seam objects)  
- `META`   — provenance/annotations (non-semantic; not executable)

### 2.3 Stability rule (hard)
For Triaxis v1.x:
- GHLL **IDs never change meaning** once published.
- If meaning must change, create a new ID or bump path with version suffix:
  - `N:INV.pde.div_free.v2` (new meaning)
  - existing `N:INV.pde.div_free` remains valid and unchanged

### 2.4 Deprecation rule (allowed)
Deprecation is allowed without changing semantics:
- Add metadata: `"status":"deprecated"` and `"replaced_by":"N:..."`.
- Compilers may warn, but must still be able to interpret deprecated glyphs.

---

## 3) The GHLL contract (required fields for every lexicon entry)

Every GHLL lexicon entry **MUST** be a JSON object (or equivalent internal struct) with the following fields.

### 3.1 Required fields
- `id` (string) — the canonical GHLL ID (e.g., `N:INV.pde.div_free`)
- `kind` (enum) — one of the kinds in §2.2
- `domain` (string) — canonical domain tag (e.g., `NS`, `GR_NR`, `RFE_UFE`, `ZETA`, `CONTROL`, `CORE`)
- `semantics` (string) — precise meaning in plain language (unambiguous)
- `formal` (object) — formal statement in a normalized mini-format (see §3.2)
- `type_signature` (string) — input/output types or “constraint over X”
- `scope` (object) — where/when it applies (time, region, object)
- `constraints` (object) — tolerances, bounds, admissibility conditions
- `proof_obligations` (array) — what must be proven/checked to claim compliance
- `lowering` (object) — binding requirements to Praxica (`required_map_ids` or explicit lowering class)
- `witness` (object) — what Aeonica must record to claim the intent was enforced
- `version_introduced` (string) — e.g., `"1.2"`
- `tests_required` (array) — minimal conformance tests

### 3.2 Formal statement format (normative mini-structure)
`formal` is not free-form LaTeX. It is a structured intent statement so tools can reason about it.

Allowed `formal.kind` values:
- `equation` (lhs/rhs)
- `inequality` (lhs/op/rhs)
- `predicate` (name/args)
- `set_membership` (elem/in_set)
- `functional` (name/definition/variables)

Example (divergence-free):
```json
"formal": {
  "kind": "equation",
  "lhs": "div(v)",
  "rhs": "0"
}
````

Example (tail barrier):

```json
"formal": {
  "kind": "inequality",
  "lhs": "S_j_max",
  "op": "<=",
  "rhs": "c_star"
}
```

---

## 4) Hashing + ledger identity (intent_hash is mandatory for truth)

### 4.1 Canonical hash function

GHLL entries are hashed into `intent_hash` so receipts can bind truth to meaning.

* Hash function: **SHA-256**
* Encoding: UTF-8
* Output: lowercase hex (64 chars)

### 4.2 Canonical serialization for hashing

To compute `intent_hash`:

1. Construct the GHLL entry object including **all fields in §3.1** except any runtime-only fields.
2. Serialize as canonical JSON:

   * keys sorted lexicographically at every object level
   * arrays preserved in order (no sorting)
   * no whitespace outside JSON tokens
   * numbers in constraints stored as decimal strings (recommended; required if you want perfect cross-language determinism)
3. Compute `sha256(canonical_json_bytes)`.

### 4.3 Ledger rule (hard)

Every `A:RCPT.step.*` receipt **MUST** include:

* `intent_id`
* `intent_hash`

If the receipt lacks either, it is **not evidence-grade**.

---

## 5) Executability + the compiler seam (how GHLL becomes action)

### 5.1 Executability rule

A GHLL glyph is **executable** if and only if:

* it is of kind `INV`, `GOAL`, `POLICY`, or `SPEC`, and
* it has at least one valid lowering route declared in `lowering`.

### 5.2 Lowering binding options (exactly one must be satisfied)

#### Option A — Required map IDs (recommended)

The entry lists required lowering maps:

* `lowering.required_map_ids = ["N:MAP....", ...]`

#### Option B — Lowering class (for compiler-owned families)

The entry references a compiler-known lowering class:

* `lowering.lowering_class = "inv.div_free.leray_projector"`

Lowering classes must be documented in the compiler pipeline doc and treated as stable APIs.

### 5.3 Seam completeness rule (hard)

If a run claims it enforced an intent `N:INV.*`, then the execution path must be witnessable as:

* `intent_id` present in receipt, and
* `ops[]` includes the opcodes implied by the active `N:MAP.*` (or by the lowering class).

If not, the claim is invalid.

---

## 6) Witness contract (what Aeonica must record for GHLL)

### 6.1 Witness fields (minimum)

Every GHLL entry must declare `witness.must_record`, which must include:

* `intent_id`
* `intent_hash`
* at least one measurable artifact of enforcement (e.g., residual, margin, gate status)

Example:

```json
"witness": {
  "must_record": ["gate_outcome", "residual:eps_div"],
  "acceptance_rule": "gate_pass && eps_div <= tol.abs"
}
```

### 6.2 Acceptance rules are declarative

`acceptance_rule` is a boolean expression over receipt fields, using:

* `gate_pass`, `gate_fail`
* `residual:<name>`
* `metric:<name>`
* `tol.abs`, `tol.rel` (from constraints)

No runtime side effects are allowed here.

---

## 7) Lexicon addition procedure (how GHLL grows safely)

### 7.1 Addition steps (mandatory)

To add a new GHLL entry:

1. **Choose stable ID** using namespace rules (§2.1).
2. **Write full lexicon entry** satisfying §3 (no missing fields).
3. **Define proof obligations** that are either:

   * statically checkable (type admissibility), or
   * dynamically checkable via Praxica + receipts (residuals/gates).
4. **Bind lowering**:

   * create the required `N:MAP.*` objects (or a lowering class)
   * ensure resulting Praxica sequences include receipt emission.
5. **Define witness acceptance rule** (§6).
6. **Provide tests** (§9).
7. **Bump GHLL lexicon version** (run manifests and receipts must record it).

### 7.2 Hard prohibitions

* No silent meaning edits to existing IDs.
* No “informal-only” intents that cannot be witnessed.
* No executable intent without a lowering path.
* No policies that contradict determinism unless explicitly marked as forbidden in Triaxis v1.x.

---

## 8) Default GHLL Lexicon Additions (Core Pack v1.2)

This section defines an executable minimum set beyond the base list, with full entries.
These are real, complete entries (not templates).

### 8.1 `N:INV.pde.div_free` (divergence-free constraint)

```json
{
  "id": "N:INV.pde.div_free",
  "kind": "INV",
  "domain": "NS",
  "semantics": "Velocity field must remain divergence-free at the enforced scope; violations trigger gate failure or corrective projection per lowering policy.",
  "formal": {"kind": "equation", "lhs": "div(v)", "rhs": "0"},
  "type_signature": "constraint over field:v (vector field on lattice)",
  "scope": {"where": "field:v", "when": "each_step", "region": "entire_domain"},
  "constraints": {"tol": {"abs": "0.000000000100", "rel": "0.000000010000"}, "norm": "L2"},
  "proof_obligations": ["compute residual eps_div", "apply projection if policy requires", "witness gate outcome"],
  "lowering": {"required_map_ids": ["N:MAP.inv.div_free.v1"]},
  "witness": {
    "must_record": ["gate_outcome:clock_stage_coherence", "residual:eps_div"],
    "acceptance_rule": "residual:eps_div <= tol.abs"
  },
  "version_introduced": "1.2",
  "tests_required": ["receipt_contains_intent_hash", "replay_identical", "div_free_projection_reduces_residual"]
}
```

### 8.2 `N:INV.pde.energy_nonincreasing` (gated energy monotonicity)

```json
{
  "id": "N:INV.pde.energy_nonincreasing",
  "kind": "INV",
  "domain": "NS",
  "semantics": "Within a declared gate scope, the chosen energy functional must not increase beyond tolerance; failure rejects the step.",
  "formal": {"kind": "inequality", "lhs": "E(t+dt)-E(t)", "op": "<=", "rhs": "0"},
  "type_signature": "constraint over functional:E(field_state)",
  "scope": {"where": "field_state", "when": "each_step", "region": "entire_domain"},
  "constraints": {"tol": {"abs": "0.000000001000", "rel": "0.000000010000"}, "functional": "kinetic_energy"},
  "proof_obligations": ["compute E_before and E_after deterministically", "witness eps_energy and gate outcome"],
  "lowering": {"required_map_ids": ["N:MAP.inv.energy_nonincreasing.v1"]},
  "witness": {
    "must_record": ["gate_outcome:energy_identity", "residual:eps_energy"],
    "acceptance_rule": "gate_outcome:energy_identity == pass && residual:eps_energy <= tol.abs"
  },
  "version_introduced": "1.2",
  "tests_required": ["receipt_contains_gate_outcomes", "reject_on_energy_increase", "replay_identical"]
}
```

### 8.3 `N:INV.clock.stage_coherence` (stage-time consistency)

```json
{
  "id": "N:INV.clock.stage_coherence",
  "kind": "INV",
  "domain": "CORE",
  "semantics": "All sub-operators in a step must evaluate at consistent stage times; mismatches are treated as coherence failure.",
  "formal": {"kind": "predicate", "name": "stage_coherent", "args": ["t_stage", "operators"]},
  "type_signature": "constraint over clocked evaluation graph",
  "scope": {"where": "operator_evals", "when": "each_step", "region": "entire_domain"},
  "constraints": {"tol": {"abs": "0.000000000000", "rel": "0.000000000000"}, "mode": "exact"},
  "proof_obligations": ["record delta_stage_t", "fail if any operator uses inconsistent stage time"],
  "lowering": {"required_map_ids": ["N:MAP.inv.clock.stage_coherence.v1"]},
  "witness": {
    "must_record": ["gate_outcome:clock_stage_coherence", "metric:delta_stage_t_max"],
    "acceptance_rule": "metric:delta_stage_t_max == 0"
  },
  "version_introduced": "1.2",
  "tests_required": ["reject_on_stage_mismatch", "receipt_records_delta_stage_t", "replay_identical"]
}
```

### 8.4 `N:INV.ledger.hash_chain_intact` (ledger continuity)

```json
{
  "id": "N:INV.ledger.hash_chain_intact",
  "kind": "INV",
  "domain": "CORE",
  "semantics": "Every receipt must satisfy sha256 canonical hashing and must chain correctly via hash_prev; any violation invalidates the run as evidence.",
  "formal": {"kind": "predicate", "name": "hash_chain_intact", "args": ["hash_prev", "hash"]},
  "type_signature": "constraint over receipt stream",
  "scope": {"where": "receipt_stream", "when": "entire_run", "region": "ledger"},
  "constraints": {"algorithm": "SHA-256", "serialization": "canonical_json_v1_2"},
  "proof_obligations": ["recompute each receipt hash", "verify hash_prev linkage", "witness verification result"],
  "lowering": {"required_map_ids": ["N:MAP.inv.ledger.hash_chain_intact.v1"]},
  "witness": {
    "must_record": ["gate_outcome:hash_chain_integrity", "metric:hash_fail_count"],
    "acceptance_rule": "metric:hash_fail_count == 0"
  },
  "version_introduced": "1.2",
  "tests_required": ["detect_hash_tamper", "replay_verifies_chain", "receipt_contains_intent_hash"]
}
```

---

## 9) Required tests for any GHLL lexicon addition

A runtime/compiler claiming support for a new GHLL entry must pass:

1. **Seam completeness test**

   * the GHLL intent lowers to a deterministic Praxica sequence (or declared lowering class)
2. **Witness completeness test**

   * step receipts contain `intent_id`, `intent_hash`, and all `witness.must_record` fields
3. **Acceptance rule consistency**

   * the recorded receipt data is sufficient to evaluate `witness.acceptance_rule`
4. **Deterministic replay**

   * same inputs produce identical receipts (canonical JSON bytes)
5. **Policy compliance**

   * if intent is a policy, the runtime behavior matches it and is witnessed

---

## 10) Minimal manifest fields for GHLL

Runs should include in their manifest (recommended, and required for evidence-grade runs):

* `ghll_lexicon_version` (string, e.g., `"1.2"`)
* `ghll_pack_digests` (map of pack name → sha256 digest)
* `float_policy` and `reduction_policy` (so meaning claims map to deterministic execution)
* `active_maps` (array of `N:MAP.*` IDs enabled in this run)

---

## 11) Summary (the enforcement point)

* GHLL is the meaning registry for Triaxis.
* Every executable GHLL intent must have a lowering path.
* Every enforced intent must be witnessed by Aeonica via `intent_hash`.
* Meaning can grow safely because IDs are stable, versioned, and ledger-joinable.

That is how Triaxis keeps meaning expressive without letting it become unverifiable mythology.