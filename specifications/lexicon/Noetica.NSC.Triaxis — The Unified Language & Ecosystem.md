# Noetica.NSC.Triaxis — The Unified Language & Ecosystem (v1.0 - Draft)

**Version:** v1.0
**Date:** 2026-01-27
**Scope:** Foundational principles, core components, and interrelationships within the Noetica, NSC, and Triaxis (Noetica/Praxica/Aeonica) ecosystem. This document serves as the canonical "terms contract" for the system's architecture and language definitions.

---

## 1. Core Philosophy: "Meaning Performs Computation"

The entire ecosystem is built on the principle that **meaning performs computation**. This is achieved through a multi-layered, verifiable, and reproducible approach that bridges high-level symbolic intent with low-level deterministic execution and an immutable audit trail. The core aims are **reproducibility, verifiability, and auditability** of computational processes.

## 2. The Triaxis (NPA) Framework

The Triaxis (NPA) is the fundamental architectural framework, decomposing the computational process into three interconnected planes, each with its own "alphabet" (glyph dialect) and distinct role:

*   **Noetica (N):** The plane of **Meaning / Intent**.
*   **Praxica (P):** The plane of **Execution / Logic**.
*   **Aeonica (A):** The plane of **Memory / Witness / Time**.

**Canon Maxim:** "Two alphabets, one compiler seam, one ledger truth." This refers to:
*   **GHLL** (Noetica alphabet) and **GLLL** (Praxica alphabet).
*   The **NSC compiler** acts as the "seam" translating Noetica intent to Praxica execution.
*   **GML** (Aeonica alphabet) provides the "ledger truth".

---

## 3. The Glyphic Languages (Alphabets)

Each plane of the Triaxis is underpinned by a specific Glyphic Language, defined by its lexicon, contract, and role:

### 3.1 GHLL (Glyphs for High-Level Lexicon - Noetica)
*   **Role:** The **"meaning registry"**. Defines **what must be true** or the intent of a computation. It is **declarative** and **meaning-first**.
*   **Purpose:** Registers semantic meaning, constraints, goals, policies, and proof obligations. Dictates how these concepts lower into GLLL actions and how they *must* be witnessed by Aeonica receipts.
*   **Format:** Globally unique IDs `N:<KIND>.<PATH>` (e.g., `N:INV.pde.div_free`, `N:GOAL.optimization.energy_min`).
*   **Stability Contract:** New meanings require versioning; existing `N:*` IDs must not be overwritten.
*   **Key Concepts:**
    *   **Data Layers:** Lexicon, Semantic, Harmonic, Field, Memory.
    *   **Coherence:** Axiom of Coherence, Coherence Metric C(t), Coherence Index (Ξ), Coherence Current (Jᶜμ).
    *   **Lagrangian:** The mathematical foundation (`ℒ`) from which Equations of Motion (EOM) are derived, representing the core physical/mathematical intent.
    *   **Noetica Transformation Operator (T_N):** Maps symbolic input to semantic wavefunction.
*   **Relationship to NSC/NLLC:** GHLL defines the high-level specifications and intents that NLLC programs embody. The NSC compiler aims to translate these intents into executable forms while preserving the declared meaning and constraints.

### 3.2 GLLL (Glyphs for Low-Level Lexicon - Hadamard / Praxica-H)
*   **Role:** The **"opcode semantics registry"**. Defines **how things happen** via atomic, deterministic actions. It is **mechanism-first** (execution truth), **deterministic**, and **audit-friendly**.
*   **Purpose:** Specifies the semantics, arguments, effects, and **receipt obligations** for each opcode. Provides action glyph identity for deterministic execution.
*   **Format:** Stable opcode identities (e.g., `H64:r00..r63`).
*   **Stability Contract:** Opcode semantics are stable across v1.x; backward compatibility rules are crucial across Hadamard orders (H64 → H128). Explicitly defines "what Aeonica receipts **must** witness for GLLL execution."
*   **Key Concepts:**
    *   **Bytecode (NSC):** Minimal opcode stream emitted from glyph tokens, forming the basis of the PDE template assembly.
    *   **Opcode Table (Phase-1):** Minimal bytecode opcodes (0x01–0x08) mapping to glyphs and stack/effect semantics.
*   **Relationship to NSC/NLLC:** The NSC compiler's `nsc_to_hadamard.py` module translates the NSC's Program Intermediate Representation (PIR) into Hadamard bytecode. This bytecode *is* the concrete, executable manifestation of GLLL. The compiled output directly adheres to the GLLL contract.

### 3.3 GML (Glyphs for Memory Ledger - Aeonica)
*   **Role:** The **"witness event registry"**. Provides **append-only truth records (receipts)** for **what happened, in time**. It is the ultimate source of **"ledger truth"**.
*   **Purpose:** Records all significant events, execution traces, multi-clock receipts, and audit-grade evidence. Ensures replayability and deterministic validation of runs.
*   **Ledger Axiom:** "Truth ≡ hash-chained receipts (replayable, deterministic)."
*   **Format:** Event types `A:KIND.PATH` mapping to required fields, hash rules, and replay obligations.
*   **Stability Contract:** Lexicon additions must specify required fields, hash rules, and replay obligations. Strict immutability and hash-chain requirements are fundamental.
*   **Minimal "Truth Bundle" (per step):** For any accepted step, a GML record must allow an auditor to determine:
    *   `intent_id`, `intent_hash`: What intent was claimed?
    *   `ops[]`: What actions ran (with args digests)?
    *   `t,dt,tau,dtau`: Under what clocks?
    *   `gates{}`: Under what gates (outcomes)?
    *   `residuals{}`, `metrics{}`: What evidence was checked?
    *   `status`: Why was it accepted (derivable from witness rules)?
    *   `hash_prev`, `hash`: Is it chained and untampered?
    *   *If any cannot be answered, the run is not evidence-grade.*
*   **Key Concepts:**
    *   **Ω-ledger:** The tamper-evident receipt chain, ensuring audit-grade evidence through hashing and canonical serialization.
    *   **PhaseLoom:** The scheduler/braider that splits a single run trace into multiple threads (channels) with coupling rules, critical for multi-clock receipts and causality tracking. (27 threads in v1.x).
    *   **Coherence Gates:** Acceptance/reject logic for steps/updates based on coherence metrics.
    *   **K-resource / Coherence Budget:** Treats stability margin as a spendable "resource."
*   **Relationship to NSC/NLLC:** GML receipts are the output of executing the Hadamard bytecode generated by NSC/NLLC. They provide the verifiable proof that the execution (GLLL) satisfied the intent (GHLL).

---

## 4. Noetica Symbolic Code (NSC) & NLLC

**NSC (Noetica Symbolic Code):**
*   **Role:** The **reversible symbolic computing language** built on glyph operators. It is the "compiler seam" that translates high-level Noetica intent into executable Praxica forms.
*   **Execution Cycle:** Canonical cycle involves glyph tokenization → bytecode generation → PDE template assembly → symbolic representation export (Phase-1).
*   **Compilation Flow:**
    1.  **Input Layer:** Glyph source (e.g., NLLC programs, Chord-Field Statements).
    2.  **Parser:** Tokenizes text into semantic tensors, handling context weighting and coherence scoring.
    3.  **AST (Abstract Syntax Tree):** Structured representation of source after parsing.
    4.  **IR (Intermediate Representation):** Multiple abstraction levels (including SSA form) between AST and target code, enabling optimization.
    5.  **PIR (Program Intermediate Representation):** The NSC-specific IR that is flattened and translated.
    6.  **`nsc_to_hadamard.py`:** Crucial module that translates PIR into **Hadamard Bytecode (GLLL)**.
    7.  **Target Compilation:** For Phase-2, potential compilation to LLVM IR for solver/renderer integration.

**NLLC (Noetica Logic Language for Computation):
*   **Role:** The file/spec layer for **"meaning performs computation" programs**.
*   **Purpose:** Primarily used for compliance tests and defining auditable execution workflows based on Noetica principles. NLLC programs implicitly or explicitly leverage GHLL definitions to guide their computation.

---

## 5. Runtime Environment & Validation

*   **GM-OS (Glyph Manifold OS):** The overarching runtime environment and visualizer that hosts Noetica modules and field rendering paths.
*   **Field Interpreter (NFI):** Simulates field equations from glyphic statements.
*   **Resonance Synth (RSE):** Converts symbolic phrases into harmonic audio fields for acoustic coherence experiments.
*   **Live Renderer (Phase-2):** Visualizes field dynamics (e.g., FFT(θ) → spectral centroid/entropy + phase-map visualization).
*   **Validation Pipeline (v7):** A critical stage flow: Glyph sequence → Resonant packet → θ-grid → Coherence Metric C(t) → Output. Failure (e.g., C < 0.995) triggers GM-OS error reports.
*   **Rails-only control:** A crucial constraint ensuring the runtime acts only through pre-declared "safe" control channels, preventing hidden state edits.

---

## 6. Key Interactions & Data Flow

1.  **Intent Definition (GHLL):** High-level goals, invariants, and policies are established using the GHLL lexicon.
2.  **Program Formulation (NLLC/Noetica Statements):** Developers write programs or symbolic statements that express GHLL intents.
3.  **Compilation (NSC - "Compiler Seam"):**
    *   The NSC parser processes NLLC/Noetica input.
    *   An AST is generated, followed by various IR stages (PIR).
    *   `nsc_to_hadamard.py` translates PIR into executable **Hadamard Bytecode (GLLL)**, ensuring compliance with GLLL contracts (semantics, arguments, effects, receipt obligations).
4.  **Execution (Praxica-H):** The Hadamard Bytecode is executed by the runtime.
5.  **Witnessing (GML):** As the bytecode executes, the runtime generates **GML receipts** for every significant step. These receipts meticulously record:
    *   The originating `intent_id` and `intent_hash`.
    *   The specific `ops[]` (GLLL opcodes) that ran.
    *   Clock information, gate outcomes, and evidence (`residuals{}`, `metrics{}`).
    *   A `status` explaining acceptance/rejection.
    *   Cryptographic hash chaining (`hash_prev`, `hash`) to ensure immutability and auditability (the Ω-ledger).
6.  **Validation & Audit:** GML receipts are continuously checked for schema validity, hash continuity, and seam join keys to ensure "evidence-grade truth". Coherence metrics are evaluated against GHLL-defined thresholds, with failures triggering GM-OS reports.

This entire system forms a closed loop, ensuring that every computation, from its highest-level meaning to its lowest-level execution, is transparent, verifiable, and leaves an immutable, auditable trail.
```