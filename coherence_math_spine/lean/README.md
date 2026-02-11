# Coherence Framework Lean Formalization (Canonical)

The canonical Lean 4 formalization of Coherence Framework theorems is in the [`coherence-theorems/`](./coherence-theorems/) subdirectory.

## Organization

- **[`coherence-theorems/`](./coherence-theorems/)** – Active Lean 4 formalization (production)
  - [`README.md`](./coherence-theorems/README.md) – Active proofs and development roadmap
  - Built and tested in CI/CD pipeline

- **[`PROOF_STATUS.md`](./PROOF_STATUS.md)** – Current formalization progress and theorem status
  - Tracks which theorems are formalized, in progress, or planned

## Canonical Home

Per [ADR-0001-lean-canonical-home.md](../../docs/adr/ADR-0001-lean-canonical-home.md), this directory (`coherence_math_spine/lean/`) is the **canonical home** for Coherence Framework's Lean formalization.

The experimental directory at [`../../../lean/README.md`](../../../lean/README.md) (`lean/NoeticanLabs/`) contains exploratory work and is **not built in CI**.

## Getting Started

For active development and theorem formalization work:
1. Navigate to [`coherence-theorems/`](./coherence-theorems/)
2. Read [`coherence-theorems/README.md`](./coherence-theorems/README.md)
3. Check [`PROOF_STATUS.md`](./PROOF_STATUS.md) for current status and opportunities to contribute
