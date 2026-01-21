# GLLL_LEXICON_ADDITIONS_AND_CONTRACT.md
# Triaxis (NPA) — GLLL Lexicon Additions & Contract (v1.2)

**Alphabet:** GLLL (Hadamard / Praxica-H)  
**Role:** Action glyph identity for deterministic execution  
**Canon maxim:** Two alphabets, one compiler seam, one ledger truth.

This document is **normative** for:
1) how new GLLL opcodes are introduced (lexicon additions),
2) what every GLLL opcode **must** specify (contract),
3) backward compatibility rules across Hadamard orders (H64 → H128 → …),
4) what Aeonica receipts **must** witness for GLLL execution.

---

## 1) What “GLLL Lexicon” means (and what it does not)

### 1.1 GLLL lexicon = opcode semantics registry
The **GLLL lexicon** is the authoritative mapping:

\[
\texttt{op\_id} \;\;\longleftrightarrow\;\; \texttt{semantics + args + effects + receipt obligations}.
\]

- It is **mechanism-first** (execution truth), not meaning-first.
- It is **deterministic** and **audit-friendly**.

### 1.2 What GLLL is NOT
- Not a meaning language (that’s GHLL / Noetica).
- Not a memory language (that’s GML / Aeonica).
- Not “free-form extension hooks.” Every extension is a **typed, versioned contract**.

---

## 2) Baseline: H64 is the stable core (v1.x)

### 2.1 Stability rule (hard)
All opcodes `H64:r00..H64:r63` are **stable identities** in Triaxis v1.x:
- **Mnemonic and semantics must never change.**
- Deprecation is allowed only by *disuse*, never by changing meaning.

### 2.2 Core defined elsewhere
The normative H64 opcode table lives in:
- `14_TRIASIS_GLYPH_CODEBOOK_NPA.md` (GLLL section)

This document governs **how we extend** beyond that baseline.

---

## 3) Two legal extension mechanisms (choose deliberately)

Triaxis supports two extension forms—both deterministic, but with different costs.

### 3.1 Mechanism A — “Macro-ops” (recommended for most growth)
**Macro-ops** are named instruction templates built from stable H64 primitives.

- Macro-op ID format: `P:MACRO.<name>.v<major>`
- Macro-ops **do not** require new Hadamard rows.
- They compile to a fixed sequence of existing opcodes (often with parameters).

**When to use:** new domain behavior that is still expressible via existing ops  
(e.g., “RK4 stage,” “compute residual pack,” “projection+filter+check bundle”).

**Why:** keeps the LLL small, stable, and replayable.

### 3.2 Mechanism B — “New opcodes” via higher Hadamard order (H128+)
When you truly need new atomic primitives (dispatch speed, IR clarity, or strict audit boundaries),
you extend the alphabet by moving to a larger Hadamard order.

**Rule:** The next expansion order is `H128`.

- New opcode IDs are `H128:r64..H128:r127`.
- H64 opcodes remain valid and stable.
- If the runtime is operating in `H128` mode, H64 opcodes are treated as **logical opcodes** that can be expanded for channel encoding (see §4).

**When to use:** operations that must be *primitive* for performance or auditing  
(e.g., kernel-call primitive with deterministic kernel registry, multigrid restrict/prolong primitives, etc.)

---

## 4) Backward compatibility across Hadamard orders (no drift allowed)

### 4.1 Canonical codeword expansion (H64 → H128)
If a channel uses Hadamard codewords (not just row indices), define:

- Let `h` be a ±1 vector of length 64 (a row of H64).
- Define the expanded 128-bit codeword:
\[
\mathrm{expand64to128}(h) = (h,\, h)
\]
(concatenate the row with itself).

**Consequences:**
- H64 identities remain stable logical opcodes.
- Channels can use H128 robustness without changing opcode IDs.

### 4.2 Decode + witness rule
If decoding from noisy codewords, the executor **MUST**:
1) decode by correlation,
2) compute `decode_margin_min` (minimum margin across all decoded ops in the step),
3) include `decode_margin_min` in the Aeonica step receipt.

---

## 5) GLLL Lexicon Entry Contract (required fields)

Every GLLL opcode (H64 or H128) must have a lexicon entry with:

### 5.1 Required fields
- `op_id` (e.g., `H64:r53`)
- `mnemonic` (ASCII uppercase, e.g., `PROJECT`)
- `class` (one of: `CONTROL | MEMORY | MATH | RAILS | KERNEL | TIME | LEDGER`)
- `semantics` (precise description of state transition)
- `args_schema` (JSON schema-like constraints)
- `effects` (read/write/emit/gate/rollback)
- `determinism` (what must be fixed to ensure replay)
- `receipt_obligations` (what Aeonica must record when this op runs)
- `safe_mode_policy` (allowed / restricted / forbidden in safe-mode)
- `version_introduced` (e.g., `1.2`)
- `tests_required` (minimal conformance tests)

### 5.2 Normalized lexicon entry format (canonical)
```json
{
  "op_id": "H64:r53",
  "mnemonic": "PROJECT",
  "class": "RAILS",
  "semantics": "Apply projection operator named in args.project to target fields; must be deterministic and side-effect limited to declared writes.",
  "args_schema": {
    "project": {"type": "string", "enum": ["leray", "constraint", "custom_id:<hex32>"]},
    "target": {"type": "string", "default": "field:v"}
  },
  "effects": ["reads:field:v", "writes:field:v", "gate:active_if_in_scope"],
  "determinism": {
    "requires": ["fixed_float_policy", "fixed_kernel_impl", "fixed_order_of_ops"],
    "forbids": ["data_race", "nondeterministic_parallel_reduction"]
  },
  "receipt_obligations": {
    "must_record": ["op_id", "args_digest", "thread_tag_if_changed"],
    "must_update_step_receipt": true
  },
  "safe_mode_policy": "allowed",
  "version_introduced": "1.2",
  "tests_required": ["replay_identical", "gate_scope_consistency", "receipt_fields_present"]
}
```

---

## 6) Addition procedure (how new GLLL entries are accepted)

### 6.1 Lexicon addition steps (mandatory)

A new opcode or macro-op is added only if all steps succeed:

1. **Define** lexicon entry (fields in §5).
2. **Assign** a stable ID:

   * Macro-op: `P:MACRO.<name>.v<major>`
   * Opcode: `H128:r64..r127` (never reuse)
3. **Specify determinism envelope**: floating policy, reduction order, kernel IDs.
4. **Specify receipt obligations**: what must be witnessed.
5. **Provide tests**: see §9.
6. **Update compiler seam**:

   * if GHLL lowering uses it, create/update `N:MAP.*` entries with explicit op sequence.
7. **Bump lexicon version**:

   * add `glll_lexicon_version` to run manifest + receipts.

### 6.2 Hard prohibitions

* No opcode ID reuse.
* No semantic edits to existing opcode entries.
* No hidden side effects (writing to undeclared state is invalid).
* No nondeterministic behavior unless explicitly seeded and witnessed (and only if policy allows).

---

## 7) Receipt obligations (Aeonica truth requirements)

### 7.1 Step receipts must witness GLLL usage

Every `A:RCPT.step.accepted` / `A:RCPT.step.rejected` **MUST** include:

* `ops[]` (opcode IDs in execution order)
* `glll_lexicon_version` (string)
* `decode_margin_min` (int, if decoding occurred; otherwise omit)
* `op_args_digests[]` (optional but recommended; see below)

### 7.2 Args digests (recommended for forensic joins)

For each op with args, compute:

* `args_digest = sha256(canonical_json(args))`

Then include:

* `op_args_digests`: array aligned with `ops[]`

This lets audits prove what parameters were used without dumping massive args.

---

## 8) Default GLLL Lexicon Additions (Extension Pack v1.2)

This pack defines **new primitive opcodes** using `H128` rows.
If your runtime does not enable H128, you may implement these as macro-ops instead,
but **the IDs and semantics remain defined**.

### 8.1 H128 additions (r64–r95) — defined, executable

| Op ID    | Mnemonic   | Class   | Semantics (normative)                                                                     |
| -------- | ---------- | ------- | ----------------------------------------------------------------------------------------- |
| H128:r64 | KCALL      | KERNEL  | Call a registered deterministic kernel by `kernel_id`; args include `kernel_id`, `io_map` |
| H128:r65 | STENCIL    | KERNEL  | Apply named stencil operator deterministically to a field region                          |
| H128:r66 | DERIV      | KERNEL  | Compute derivative operator (∂x/∂y/∂z) with declared scheme id                            |
| H128:r67 | LAPLACE    | KERNEL  | Compute Laplacian with declared scheme id                                                 |
| H128:r68 | ADVECT     | KERNEL  | Apply advection update (scheme id must be declared)                                       |
| H128:r69 | DIFFUSE    | KERNEL  | Apply diffusion/viscosity update (scheme id must be declared)                             |
| H128:r70 | RESTRICT   | KERNEL  | Multigrid restriction (downsample) deterministic                                          |
| H128:r71 | PROLONG    | KERNEL  | Multigrid prolongation (upsample) deterministic                                           |
| H128:r72 | RESID      | LEDGER  | Compute residual pack into a declared residual buffer                                     |
| H128:r73 | NORM2      | MATH    | Compute L2 norm deterministically (fixed reduction order required)                        |
| H128:r74 | AXPY       | MATH    | y ← a*x + y (BLAS-like), deterministic                                                    |
| H128:r75 | SCALE      | MATH    | x ← a*x, deterministic                                                                    |
| H128:r76 | COPY       | MEMORY  | y ← x copy, deterministic                                                                 |
| H128:r77 | MINMAX     | MATH    | Compute min/max deterministically (fixed reduction order required)                        |
| H128:r78 | DIGEST     | LEDGER  | Compute `state_digest` for declared state slice                                           |
| H128:r79 | MANIFEST   | LEDGER  | Emit/refresh run manifest digest (must be witnessed)                                      |
| H128:r80 | TAGT       | TIME    | Attach explicit thread tag change (must be witnessed)                                     |
| H128:r81 | DT_ARB     | TIME    | dt arbitration primitive (sets dt per policy, must be witnessed)                          |
| H128:r82 | TAU_ARB    | TIME    | dtau arbitration primitive (sets dtau per policy, must be witnessed)                      |
| H128:r83 | FAILFAST   | RAILS   | Immediate reject-step trigger with reason (must emit rejected receipt)                    |
| H128:r84 | SOFTFAIL   | RAILS   | Mark a soft failure; step may continue but must emit WARN                                 |
| H128:r85 | MEASURE    | LEDGER  | Record named metric into receipt metrics (deterministic source required)                  |
| H128:r86 | TRACE      | LEDGER  | Emit a lightweight trace event (non-step receipt)                                         |
| H128:r87 | LIMIT      | RAILS   | Apply policy limit (e.g., clamp dt, clamp residual growth), deterministic                 |
| H128:r88 | SCHEDULE   | CONTROL | Deterministic scheduler hint (yield/priority within policy)                               |
| H128:r89 | TOKEN_NEW  | CONTROL | Create a deterministic token handle for synchronization                                   |
| H128:r90 | TOKEN_WAIT | CONTROL | Wait for token deterministically under scheduler policy                                   |
| H128:r91 | TOKEN_SET  | CONTROL | Set token state deterministically                                                         |
| H128:r92 | PACKF      | MEMORY  | Pack field slice to contiguous buffer (deterministic layout)                              |
| H128:r93 | UNPACKF    | MEMORY  | Unpack buffer to field slice (deterministic layout)                                       |
| H128:r94 | BARRIER    | CONTROL | Global barrier in deterministic order                                                     |
| H128:r95 | FENCE      | CONTROL | Memory fence / visibility boundary                                                        |

### 8.2 H128 reserved (r96–r127) — defined as illegal in v1.2

These IDs are **RESERVED** and their semantics are:

* **Illegal to execute** (`TRAP` equivalent)
* **Must be rejected at validation time** (compiler or runtime)
* If observed in a receipt: run is invalid evidence

Reserved IDs:

* `H128:r96` through `H128:r127` are **RESERVED_ILLEGAL_v1_2**

This is not a placeholder; it is a *hard safety definition*.

---

## 9) Conformance tests required for any lexicon addition

A runtime claiming support for a new GLLL entry must pass:

1. **Deterministic replay test**

   * Same input → identical receipts (byte-for-byte canonical JSON)
2. **Receipt obligation test**

   * Step receipts contain `ops[]`, `glll_lexicon_version`, and any op-required metrics/digests
3. **Gate interaction test**

   * If op is `RAILS` or gate-affecting, gate outcome fields appear and behave as specified
4. **Safe-mode compliance**

   * In safe-mode, forbidden ops reject; allowed ops run; all witnessed
5. **Validation rejection**

   * RESERVED_ILLEGAL opcodes are rejected before execution

---

## 10) Minimal fields to add to run manifest + step receipts

### 10.1 Run manifest (recommended)

* `glll_base_order`: `"H64"` or `"H128"`
* `glll_lexicon_version`: `"1.2"`
* `float_policy`: `{ "type":"IEEE-754", "precision":"binary64", "round":"nearest_even" }`
* `reduction_policy`: `{ "order":"fixed", "tree":"deterministic" }`
* `kernel_registry_digest`: `"sha256:<hex>"` (if KCALL/STENCIL/etc are used)

### 10.2 Step receipts (required when relevant)

* `glll_lexicon_version`
* `decode_margin_min` (if decoding occurred)
* `state_digest` (if `DIGEST` ran)
* `kernel_id` digests (if `KCALL` ran, either in args digests or explicit metrics)

---

## 11) Summary (the enforcement point)

* **GLLL is small, strict, and stable.**
* Most growth should be **macro-ops** (templates over H64).
* True primitives expand via **H128** with immutable semantics.
* Aeonica receipts witness *what ran*, *with what parameters*, and *under what lexicon version*.

That’s how Triaxis keeps performance without sacrificing truth.