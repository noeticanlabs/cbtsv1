# Triaxis (NPA) Integration Master Plan

## Overview
This plan details the steps to integrate the **Triaxis (Noetica-Praxica-Aeonica)** specifications into the current GR Solver and NSC architecture. The goal is to ensure the runtime system strictly adheres to the v1.2 Canon for glyphs, receipts, and contracts, enabling "evidence-grade" simulation.

**Reference Documents:**
- `14_TRIASIS_GLYPH_CODEBOOK_NPA.md` (The Truth Alphabet)
- `GML_GLYPH_LIBRARY_DICTIONARY.md` (Witness Dictionary)
- `GHLL_GLYPH_LIBRARY_DICTIONARY.md` (Meaning Dictionary)
- `42_AEONICA_RECEIPTS.md` (Receipt Format)

---

## Phase 1: Lexicon Binding (The "Truth" Layer)
**Objective:** Bind internal Python symbols and strings to canonical Triaxis IDs to prevent semantic drift.

### 1.1 Create Canonical Lexicon Module
- **File:** `src/triaxis/lexicon.py`
- **Action:** Define immutable constants for all IDs in `14_TRIASIS_GLYPH_CODEBOOK_NPA.md`.
  - `GHLL_IDS`: `N:INV.pde.div_free`, `N:POLICY.deterministic_replay`, etc.
  - `GLLL_OPS`: `H64:r00` to `H64:r63` mapping.
  - `GML_THREADS`: Full 27-thread list (`A:THREAD.PHY.L.R0`, etc.).

### 1.2 Map GR Constraints to GHLL Intents
- **Target:** `gr_solver/gr_constraints.py`
- **Action:**
  - Map Hamiltonian constraint check to `N:INV.pde.energy_nonincreasing` (or specific GR equivalent).
  - Map Momentum constraint check to `N:INV.pde.div_free` (conceptual mapping) or create new `N:INV.gr.*` if needed (via extension contract).
  - Ensure `compute_residuals` returns dicts keyed by these canonical IDs.

---

## Phase 2: Aeonica Witness Layer (The "Memory" Layer)
**Objective:** Upgrade receipt emission to meet GML v1.2 strict schema and hashing rules.

### 2.1 Canonical JSON Serialization
- **Target:** `aeonic_receipts.py`
- **Action:**
  - Implement `canonical_json_dumps(obj)`:
    - Keys sorted lexicographically.
    - No whitespace.
    - **Crucial:** Floats formatted as decimal strings (e.g., `"0.010000000"`) to ensure hash determinism across platforms.

### 2.2 Hash Chain Implementation
- **Target:** `aeonic_receipts.py`
- **Action:**
  - Add `hash_prev` field to every receipt.
  - Implement SHA-256 hashing of the canonical JSON string (excluding `hash` field).
  - Verify chain continuity in `AeonicMemoryBank`.

### 2.3 Mandatory Fields Injection
- **Target:** `gr_solver/gr_ledger.py`
- **Action:** Ensure every `A:RCPT.step.*` includes:
  - `intent_id`: The GHLL ID of the primary physics model (e.g., `N:SPEC.ns_cpu_gold.v1` or GR equivalent).
  - `ops[]`: List of GLLL opcodes executed (can be high-level summary if full trace unavailable, but must be valid IDs).
  - `thread`: The active PhaseLoom thread ID.

---

## Phase 3: PhaseLoom 27-Thread Lattice (The "Control" Layer)
**Objective:** Replace ad-hoc threads with the canonical 27-thread taxonomy.

### 3.1 Implement 27-Thread Structure
- **Target:** `phaseloom_threads_gr.py` (or `phaseloom_27.py`)
- **Action:**
  - Define the 3x3x3 lattice:
    - **Domains:** `PHY`, `CONS`, `SEM`
    - **Scales:** `L` (Low/Macro), `M` (Mid/Cascade), `H` (High/Tail)
    - **Responses:** `R0` (Observe), `R1` (Control/Damp), `R2` (Audit/Rollback)
  - Map current residuals to these bins (e.g., `eps_H` -> `CONS.M`).

### 3.2 Dominance Arbitration
- **Target:** `gr_scheduler.py`
- **Action:**
  - Implement `arbitrate_dt()` using the 27 threads.
  - Logic: `dt = min(dt_cap(thread) for thread in threads)`.
  - Identify `dominant_thread` for the receipt.

---

## Phase 4: Praxica/GLLL Alignment (The "Action" Layer)
**Objective:** Ensure the NSC/Hadamard execution layer uses standard opcodes.

### 4.1 Opcode Mapping
- **Target:** `src/hadamard/assembler.py`
- **Action:**
  - Align internal opcode constants with `14_TRIASIS_GLYPH_CODEBOOK_NPA.md`.
  - Example: `GATE_B` must be `H64:r48` (0x30).
  - Ensure `SEAL` (`H64:r61`) is called at the end of every step.

### 4.2 Gate Semantics
- **Target:** `src/hadamard/vm.py`
- **Action:**
  - Ensure `GATE_B` / `GATE_E` scopes enforce `CHECK` obligations.
  - Implement `SAFE` (`H64:r63`) mode triggers.

---

## Phase 5: Contract Enforcement & Verification
**Objective:** Validate the integration using GCAT tests.

### 5.1 SEM-Hard Failures
- **Target:** `gr_solver/gr_phaseloom_orchestrator.py`
- **Action:**
  - Enforce `Orchestrator Contract`: If `accepted_history` is insufficient, refuse verification.
  - Enforce `Solver Contract`: If prerequisites (Christoffels) missing, raise `SEM_FAILURE`.

### 5.2 Verification Suite
- **Target:** `tests/test_triaxis_compliance.py`
- **Action:** Create a new test suite that:
  1. Runs a short simulation.
  2. Inspects the `receipts.jsonl`.
  3. Verifies:
     - All IDs are valid `A:*`, `N:*`, `H:*`.
     - Hash chain is unbroken.
     - Floats are strings.
     - Every step has a valid 27-thread ID.