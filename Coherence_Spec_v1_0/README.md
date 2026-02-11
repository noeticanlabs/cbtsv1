# Coherence Spec v1.0

## Intent

Coherence in this specification is an **operational, measurable, auditable governance discipline**. It is defined as a *governance-measurable integrity budget* consisting of a scalar debt \(\mathfrak C\) plus a residual vector \(\mathbf r\). Systems declare gates and rails, emit receipts, and enforce bounded correction.

**Not included:** This spec makes **no metaphysical or physical claims** unless they are explicitly labeled **BRIDGE/UNVERIFIED** in the bridge notes. Coherence here is an *affordability condition* for steps and claims, not a statement of ultimate truth.

## Navigation

- [01 Coherence Principle](docs/01_Coherence_Principle.md)
- [02 Formal Definitions](docs/02_Formal_Definitions.md)
- [03 Axioms and Postulates](docs/03_Axioms_and_Postulates.md)
- [04 Coherence Functionals](docs/04_Coherence_Functionals.md)
- [05 Coherence Transport and Balance](docs/05_Coherence_Transport_and_Balance.md)
- [06 Gates, Rails, and Affordability](docs/06_Gates_Rails_and_Affordability.md)
- [07 Coherence Lemmas (Operational)](docs/07_Coherence_Lemmas_Operational.md)
- [08 Algorithms (Canonical)](docs/08_Algorithms_Canonical.md)
- [09 Receipts, Ledgers, and Certificates](docs/09_Receipts_Ledgers_and_Certificates.md)
- [10 Verification and Test Plan](docs/10_Verification_and_Test_Plan.md)
- [11 Failure Modes and Recovery](docs/11_Failure_Modes_and_Recovery.md)
- [12 Glossary and Lexicon](docs/12_Glossary_and_Lexicon.md)
- [13 Bridge Notes (Time/GR/PDE/AI)](docs/13_Bridge_Notes_Time_GR_PDE_AI.md)
- [14 Coherence Governance Kernel](docs/14_Coherence_Governance_Kernel.md)
- Schemas: [coherence receipt](schemas/coherence_receipt.schema.json), [gate policy](schemas/gate_policy.schema.json), [run certificate](schemas/run_certificate.schema.json)
- Examples: [receipt](examples/receipt.example.json), [gate policy](examples/gate_policy.example.json), [certificate](examples/certificate.example.json)
- [Manifest](manifest/manifest.md)

## Compliance Checklist (minimum requirements)

A system may claim **Coherence-compliant** only if it satisfies **all** of the following:

1. **Computes coherence**: emits a residual vector \(\mathbf r\) and scalar debt \(\mathfrak C\) per step.
2. **Enforces gates/rails**: hard gates must pass; soft policy must hold; rails are bounded.
3. **Produces receipts**: receipts are replayable and include gate decisions, residuals, debt decomposition, and bounded actions.
4. **Bounded correction**: corrective actions are bounded and logged.
5. **Deterministic replay**: same inputs/config/seed produce equivalent receipts within declared tolerances.

### Core statement (minimum)

Coherence is a **governance-measurable integrity budget** consisting of a scalar debt \(\mathfrak C\) and a residual vector \(\mathbf r\), with gates, rails, and receipts. A system is **coherence-compliant** only if it produces replayable receipts and enforces bounded correction.
