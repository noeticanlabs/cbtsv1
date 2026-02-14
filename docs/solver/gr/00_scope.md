# GR Solver Scope

## Mission

CBTSV1 is a purpose-built General Relativity (GR) solver + framework repository. It contains:

1. **The actual GR/RFT solver code** (and numerics needed by it)
2. **Framework runtime glue** needed to run/validate it (receipts, minimal conformance)
3. **Documentation** explaining how to run the solver and what equations it implements
4. **Tests and small example datasets**

## Formulations Implemented

The solver supports multiple formulations of Einstein's equations:

- **ADM** (Arnowitt-Deser-Misner) - 3+1 decomposition
- **BSSN** (Baumgarte-Shapiro-Shibata-Nakamura) - conformal traceless formulation
- **Generalized Harmonic** (future)

## Target Spacetimes

- Minkowski (flat spacetime - validation)
- Schwarzschild (single black hole)
- Kerr (rotating black hole - future)

## Not in Scope

This repository does **not** contain:

- Language compilers (NSC, NLLC) - see archived/non-solver ecosystems
- General-purpose governance frameworks - only minimal runtime
- Duplicate documentation spines - see archive/docs_legacy
- Generated outputs - see artifacts/
