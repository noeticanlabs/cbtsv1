# 30_PRAXICA_SPEC.md
# Praxica (P) — Execution Language Spec (v1.2)

**Role:** Praxica is Triaxis’s deterministic execution layer.  
**Alphabet:** GLLL (Hadamard-coded opcode identities).  
**Prime directive:** Rails-first execution with ledger-grade witnessing.

---

## 1) Execution model

### 1.1 Program structure
A Praxica program is a set of blocks:
- Each block is a sequence of **instructions**
- Control flow is explicit via `JMP`, `BR`, `CALL`, `RET`, `LOOP_B`, `LOOP_E`

### 1.2 State model
The machine state consists of:
- **Registers:** fixed set `r0..rN` (implementation-defined count; must be recorded in run manifest)
- **Memory regions:** arena/stack allocations via `ALLOC`/`FREE`
- **Clock state:** `(t, dt)` and coherence clock `(tau, dtau)`
- **Gate state:** active gate scopes and their parameters
- **Ledger state:** `hash_prev` head and receipt buffer

---

## 2) Opcode identity via Hadamard (GLLL)

### 2.1 Hadamard matrix generation (normative)
For order `n = 2^k`, generate \(H_n\) by Sylvester recursion:

\[
H_1=[1],\quad
H_{2n}=\begin{bmatrix}
H_n & H_n \\
H_n & -H_n
\end{bmatrix}
\]

Row `i` is the codeword for opcode `Hn:ri`.

### 2.2 Encoding and decoding
Represent a received codeword \(y\in\{\pm1\}^n\). Decode by correlation:

\[
\hat{i}=\arg\max_i \langle y, h_i\rangle
\]

**Robustness:** minimum Hamming distance between distinct rows is \(n/2\).  
Unique decode is guaranteed if bit flips \(< n/4\).

### 2.3 Dispatch identity rule
- Runtime dispatch **MUST** use the opcode ID `H64:rXX`.
- If input is received as a noisy codeword, the system **MUST**:
  1) decode to `H64:rXX` by correlation,
  2) record `decode_margin_min` in receipts,
  3) reject execution if margin is below policy threshold (policy lives in Noetica).

---

## 3) Instruction format

### 3.1 Logical instruction object (normalized)
```json
{
  "op": "H64:r53",
  "mode": {
    "predicated": false,
    "vectorized": false,
    "safe_only": true
  },
  "args": {
    "project": "leray"
  },
  "effects": [
    "reads:field:v",
    "writes:field:v"
  ]
}
```

### 3.2 Determinism requirements

A Praxica implementation **MUST** be deterministic under replay, meaning:

* identical inputs + identical initial state + identical policies → identical receipts
* any randomness must be explicit (seeded) and included in receipts
* floating behavior must be declared in the run manifest:

  * IEEE-754 binary64
  * rounding mode: `nearest_even`
  * flush-to-zero policy: declared and consistent across run

---

## 4) Gates (rails-first execution)

### 4.1 Gate scopes

Gate scopes are opened/closed by:

* `H64:r48` (GATE_B)
* `H64:r49` (GATE_E)

Inside a gate scope, additional obligations apply:

* mandatory checks (`CHECK`) at specified points
* mandatory emits (`EMIT`) at least once per step if policy requires

### 4.2 Standard gates (minimum set)

Praxica must support these gate names as canonical strings:

* `energy_identity`
* `tail_barrier`
* `clock_stage_coherence`
* `hash_chain_integrity`

Gate outcomes must be included in `A:RCPT.step.*` receipts.

---

## 5) κ-DM budget integration (K-resource)

### 5.1 Budget operation

Opcode `H64:r58` (BUDGET) updates coherence resource usage.

Normalized args:

```json
{"budget": {"consume": "0.005000000", "recharge": "0.000000000", "floor": "0.010000000"}}
```

Rules:

* budgets must never go below declared floor
* if floor would be violated, the step must be rejected and witnessed

---

## 6) Step lifecycle (how “a step” works)

### 6.1 Canonical step phases

Each time-step in a solver run must follow:

1. `TIME` snapshot (optional but recommended)
2. `GATE_B` open gate scope(s)
3. execute core ops (math/memory/projections/filters)
4. `CHECK` required invariants
5. `EMIT` required receipt packets
6. `GATE_E` close scope(s)
7. `SEAL` commit (accept or reject)
8. if rejected: `ROLLBACK` (if policy requires)

### 6.2 SEAL semantics (commit rule)

Opcode `H64:r61` (SEAL) determines step status:

* if any required gate failed: status = rejected
* else: status = accepted

SEAL **MUST** trigger emission of either:

* `A:RCPT.step.accepted` or
* `A:RCPT.step.rejected`

---

## 7) Safety modes

### 7.1 SAFE opcode

Opcode `H64:r63` forces safe-mode:

* tighter gate thresholds
* stricter decode margin thresholds
* reduced instruction allowance (implementation must declare safe-mode policy)

Safe-mode entry **MUST** be witnessed by a receipt.

---

## 8) Minimal conformance tests (must pass)

A Praxica implementation is v1.2 conformant if it can:

1. Execute a program that uses `GATE_B … CHECK … EMIT … SEAL`.
2. Emit a valid `A:RCPT.step.accepted` with correct join keys and hash chain.
3. Reject and rollback on a failing gate, emitting `A:RCPT.step.rejected` and `A:RCPT.rollback.executed`.
4. Replay the run and reproduce identical receipts (byte-for-byte canonical JSON).

---

## 9) Opcode reference

All opcodes and their semantics are defined in:

* `14_TRIASIS_GLYPH_CODEBOOK_NPA.md` (GLLL section, H64 table)