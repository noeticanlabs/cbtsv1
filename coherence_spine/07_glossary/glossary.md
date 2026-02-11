---
title: "Glossary (Canonical)"
description: "Canonical definitions of core coherence terms: residuals, debt, gates, rails, receipts, and hash chains"
last_updated: "2026-02-11"
authors: ["NoeticanLabs"]
tags: ["coherence", "glossary", "definitions", "terms"]
---

# Glossary (Canonical)

## Index
- [A](#a): Acceptance set, Affordability
- [B](#b): BridgeCert
- [C](#c): Certificate, Coherence, Canonical form, Contradiction
- [D](#d): Decision, Debt, Dissipation
- [G](#g): Gate, Hard gate, Soft gate
- [H](#h): Hysteresis, Hash chain
- [K](#k): Kuramoto order parameter
- [L](#l): Level of Consistency, Ledger
- [N](#n): Normalization scale
- [P](#p): Penalty term, Projection rule
- [R](#r): Rail, Receipt, Reconstruction, Replay, Residual
- [S](#s): Solver namespace, Soft gate
- [T](#t): Transport/Current
- [U](#u): Universal Field Equation

## Core Terms (L0-L2)

**Coherence** (L0) — Persistence of identity under evolution, quantified by residuals and enforced by gates. The measurable property of a system indicating its predictive integrity and consistency.

**Residual r(x)** (L1) — Measurable defect signal across five domains: physics, constraints, semantics, tools, and operations. Formal definition: the discrepancy between observed evolution and predicted evolution under the UFE.

**Debt C(x)** (L1) — Scalar functional aggregating residuals and penalties into an acceptance currency. Formally: C(x) := Σᵢ wᵢ‖r̃ᵢ(x)‖² + Σⱼ vⱼpⱼ(x), where wᵢ are weights and pⱼ are penalty terms. See [`coherence_math_spine/04_debt_functionals.md`](../../coherence_math_spine/04_debt_functionals.md).

**Gate** (L2) — Pass/warn/fail rule (with hysteresis) producing accept/reject/abort decisions. Gates partition the state space into acceptance, warning, and failure regions. See [`coherence_spine/04_control/gates_and_rails.md`](04_control/gates_and_rails.md).

**Hard gate** (L2) — Invariant that must never fail on accepted states; violation triggers rollback or abort. Examples: NaN/Inf checks, positivity constraints, domain bounds.

**Soft gate** (L2) — Quality threshold that triggers bounded repair via rails plus retry, not abort. Allows reconfigurable system behavior within safety margins.

**Rails** (L2) — Bounded corrective actions applied during repair: dt deflation (R1), rollback (R2), projection to constraint manifold (R3), and bounded gain adjustment (R4). See [`coherence_spine/04_control/gates_and_rails.md`](04_control/gates_and_rails.md).

**Receipt** (L2) — Auditable record of attempt: state, metrics, decisions, actions, hashes, and provenance. Hash-chained and immutable, serving as evidence for coherence verification. See [`coherence_spine/03_measurement/telemetry_and_receipts.md`](03_measurement/telemetry_and_receipts.md).

**Hash chain** (L2) — Tamper-evident linking of receipts via cryptographic hashing; prevents falsification of execution traces and ensures accountability.

**Projection rule** (L1) — Declared layer mapping defining which layers can communicate; illegal crossings are coherence violations. See [`spec/00_precedence.md`](../../spec/00_precedence.md).

## Extended Terms (L1-L3)

### A

**Acceptance set 𝒜** (L1) — The set of states satisfying all hard gate invariants and soft gate thresholds: 𝒜 := {x ∈ X : I_hard(x) = true ∧ qₗ(x) ≤ τₗ ∀ℓ}. Formally defined in [`coherence_math_spine/01_notation.md`](../../coherence_math_spine/01_notation.md). *Related: Gate, Hard gate, Soft gate.*

**Affordability** (L2) — Governance condition determining whether a step can be accepted based on available coherence budget and system constraints. Defined formally in [`coherence_spine/04_control/gates_and_rails.md`](04_control/gates_and_rails.md). *Related: Debt, Budget.*

### B

**BridgeCert** (L1) — Bridge certificate connecting physics and discrete systems; certifies that discrete residuals imply analytic bounds. The only architectural location where numerical analysis is permitted. Formally defined in [`coherence_math_spine/08_certificates.md`](../../coherence_math_spine/08_certificates.md). *Related: Certificate, Universal Field Equation.*

### C

**Certificate** (L1) — Formal attestation of coherence property preservation. Includes SOS (sum-of-squares) decompositions, interval bounds, small-gain certificates, trace certificates, and BridgeCerts. See [`coherence_math_spine/08_certificates.md`](../../coherence_math_spine/08_certificates.md).

**Canonical form** (L1) — Standard representation for comparisons and proofs; enables unique identification and consistent reasoning across the system. Used in receipt canonicalization and state equivalence checks.

**Contradiction** (L2) — Residual block indicating incompatible constraints that cannot be simultaneously satisfied. Triggers escalation to abort or requires constraint relaxation. *Related: Residual, Constraint.*

### D

**Decision** (L2) — Outcome of gate evaluation: accept, retry, or abort. Determines whether the proposed state transition is committed, repaired, or rolled back. *Related: Gate, Verdict.*

**Dissipation** (L2) — Reduction of debt magnitude via corrections and rail actions. Represents progress toward coherence through bounded repair. *Related: Debt, Rails.*

### G

*See Core Terms section.*

### H

**Hysteresis** (L2) — Lag in system response to changing conditions; implemented in gates via separate enter/exit thresholds. Prevents oscillation at boundary conditions. Default values: warn_enter=0.75·fail_enter, warn_exit=0.60·fail_enter, fail_exit=0.80·fail_enter. See [`coherence_spine/04_control/gates_and_rails.md`](04_control/gates_and_rails.md).

*See Core Terms section for Hash chain.*

### K

**Kuramoto order parameter** (L1) — Coherence metric for coupled oscillator systems: Z = (1/N)Σₖ e^(iθₖ) = R·e^(iΦ), where R ∈ [0,1] is coherence magnitude and Φ is mean phase. Formally defined in [`runtime_reference/gates.py`](../runtime_reference/gates.py). *Related: Level of Consistency, Phase coherence.*

### L

**Ledger** (L2) — Hash-chained collection of receipts providing tamper-evident audit trails. Also known as Ω-Ledger when explicitly tracking receipt chains. See [`coherence_spine/03_measurement/telemetry_and_receipts.md`](03_measurement/telemetry_and_receipts.md). *Synonym: Ω-Ledger.*

**Level of Consistency (LoC)** (L2) — Quantitative measure of system coherence combining debt, residuals, and gate status. Ranges from 0 (incoherent) to 1 (fully coherent). Used to track long-term system health.

### N

**Normalization scale sᵢ** (L1) — Scaling factor for residual component normalization, with units matching the residual block. Enables portable thresholds: r̃ᵢ(x) := rᵢ(x)/sᵢ. Formally defined in [`coherence_math_spine/01_notation.md`](../../coherence_math_spine/01_notation.md). *Related: Residual, Debt.*

### P

**Penalty term pⱼ(x)** (L1) — Component of debt functional for domain constraints: pⱼ(x) ≥ 0. Penalizes constraint violations beyond soft gate thresholds. Formally: C(x) := Σᵢ wᵢ‖r̃ᵢ(x)‖² + Σⱼ vⱼpⱼ(x). See [`coherence_math_spine/04_debt_functionals.md`](../../coherence_math_spine/04_debt_functionals.md).

### R

*See Core Terms section for Rail, Receipt, Residual.*

**Reconstruction** (L2) — State regeneration from residuals and decisions; used to verify step acceptance and detect corruption in receipt chains. Enables deterministic replay validation.

**Replay** (L2) — Re-execution of a computation to verify coherence; reconstructs state from receipt chain to validate gate verdicts and detect inconsistencies. *Related: Receipt, Hash chain, Contradiction.*

### S

**Solver namespace** (L3) — Computational context for gate and rail operations; binds UFE operators, gate policies, and rail configurations. Enables deterministic versioning and reproducibility. Defined in receipt metadata.

### T

**Transport/Current** (L2) — Flow of coherence across system layers via the BRIDGE concept. Represents how coherence debt moves between physics (continuous) and discrete (computational) domains. See [`coherence_spine/05_runtime/reference_implementations.md`](05_runtime/reference_implementations.md). *Related: BridgeCert.*

### U

**Universal Field Equation (UFE)** (L1) — Mathematical framework unifying residual dynamics into three components: Ψ̇ = Lphys(Ψ) + Sgeo(Ψ) + ΣGᵢ(Ψ), where Lphys is physical evolution, Sgeo is geometric/gauge correction, and Gᵢ are drive operators. Canonical reference: [`canon/UFE_MASTER.md`](../../canon/UFE_MASTER.md). *Related: Residual, BridgeCert.*

## Layer Attribution

**L0 (Principle):** Coherence
**L1 (Mathematics):** Residual, Debt, UFE, BridgeCert, Certificate, Acceptance set, Normalization scale, Penalty term, Canonical form
**L2 (Specification):** Gate, Hard gate, Soft gate, Rail, Receipt, Hash chain, Projection rule, Hysteresis, Affordability, Decision, Dissipation, Reconstruction, Replay, Contradiction, Transport/Current, Level of Consistency, Solver namespace
**L3+ (Runtime):** Kuramoto order parameter

## Notation Standards

- **Debt functional:** Use C(x) everywhere (standard, concise)
- **Residual vector:** r(x) = (r_phys, r_cons, r_sem, r_tool, r_ops)
- **Normalized residual:** r̃ᵢ(x) := rᵢ(x)/sᵢ
- **Acceptance set:** 𝒜 (blackboard bold)
- **Normalization scale:** sᵢ (lowercase with subscript)
- **Penalty term:** pⱼ(x) (lowercase with subscript)
- **Kuramoto order:** R (magnitude), Φ (phase)
- **UFE operators:** Lphys, Sgeo, Gᵢ
