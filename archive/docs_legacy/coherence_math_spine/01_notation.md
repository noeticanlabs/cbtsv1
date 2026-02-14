---
title: "Notation"
description: "Standard notation for state spaces, residual maps, debt functionals, and gate sets"
last_updated: "2026-02-11"
authors: ["NoeticanLabs"]
tags: ["coherence", "notation", "mathematics", "notation-reference"]
---

# Notation

## Spaces and norms
- State space: \(X\) (metric space or normed vector space as needed).
- Metric: \(d_X(\cdot,\cdot)\).
- Norm: \(\|\cdot\|\) (context determines which; must be stated).

## Residual blocks
Residual map \(r: X \to \mathbb R^m\) partitioned into blocks:
\[
r(x) = (r_{\text{phys}}(x), r_{\text{cons}}(x), r_{\text{sem}}(x), r_{\text{tool}}(x), r_{\text{ops}}(x)).
\]

Each block may itself be vector-valued; write \(\|r_i(x)\|\) for its norm.

## Normalization scales
For each residual block choose a **scale** \(s_i>0\) with units matching \(r_i\), and define
\[
\tilde r_i(x) := r_i(x)/s_i.
\]
This makes thresholds portable.

## Debt functional (CANONICAL)
Weights \(w_i\ge 0\), penalties \(p_j(x)\ge 0\), penalty weights \(v_j\ge 0\):
\[
C(x) := \sum_i w_i\|\tilde r_i(x)\|^2 + \sum_j v_j p_j(x).
\]
\(C(x)\) is the **coherence debt** (canonical notation).

### Deprecated notation
- \(\mathfrak C(x)\) — *Deprecated* blackboard bold variant; use \(C(x)\) instead
- \(𝔉(x)\) — *Deprecated* fraktur variant; use \(C(x)\) instead

All new documents must use \(C(x)\).

## Gate sets
- Hard invariants: \(I_{\text{hard}}(x)\in\{\text{true},\text{false}\}\).
- Soft metrics: \(q_\ell(x)\in\mathbb R_{\ge 0}\) (typically selected residual norms).

Acceptance set (for a fixed policy):
\[
\mathcal A := \{x\in X : I_{\text{hard}}(x)=\text{true} \ \wedge\  q_\ell(x)\le \tau_\ell \ \forall \ell\}.
\]

## Rails (bounded repair maps)
A rail action is a (possibly state-dependent) map \(a: X\to X\) with a bound
\[
d_X(a(x),x)\le \delta_a \quad \text{for all admissible }x,
\]
and a legality condition "does not break hard invariants when invoked under its trigger".

## Capitalization Standards

### Coherence (proper noun)
Use "Coherence" (capitalized) when referring to:
- The Coherence Framework
- The Coherence Principle
- Coherence property (as defined term)
- Specific theorems like "the Coherence Existence Theorem"

**Examples:**
- ✓ "The Coherence Framework provides..."
- ✓ "Under the Coherence Principle, ..."
- ✓ "Coherence is the property that..."
- ✗ "coherence framework" (should be Coherence Framework)

### coherence (common noun)
Use "coherence" (lowercase) when referring to:
- The abstract property: "the system has coherence"
- Adjective form: "coherence value", "coherence metric", "coherence preservation"
- Measurement: "measured coherence", "quantifying coherence"

**Examples:**
- ✓ "The coherence value ranges from 0 to 1"
- ✓ "Coherence is preserved under evolution"
- ✓ "Define coherence via residuals"
- ✗ "The Coherence value" (should be "coherence value")

## Notation Standardization Guide

This section enforces consistent mathematical notation across all documents.

### Rule 1: Debt Functional
**Canonical:** \(C(x)\)
**Rule:** Use \(C(x)\) in all new documents
**Migration:** Update existing documents using deprecated forms
**Rationale:** Concise, standard, consistent with gate predicates

### Rule 2: Residual Normalization
**Canonical:** \(\tilde r_i(x)\) (tilde over r)
**Alternative:** \(r_i^{\text{norm}}(x)\) (acceptable if tilde unavailable)
**Rule:** Subscript \(i\) for domain (phys, cons, sem, tool, ops)
**Example:** \(\tilde r_{\text{phys}}(x)\)

### Rule 3: Acceptance Set
**Canonical:** \(\mathcal A\) (calligraphic A)
**Rule:** Use blackboard bold or calligraphic; do not use plain \(A\)
**Definition:** \(\mathcal A := \{x \in X : \text{all gates pass}\}\)

### Rule 4: UFE Operators
**Canonical:** \(L_{\text{phys}}\), \(S_{\text{geo}}\), \(G_i\)
**Rule:** Use subscript notation with descriptive names
**Domain:** \(L_{\text{phys}} : X \to X\), \(S_{\text{geo}} : X \to X\), \(G_i : X \to X\)

### Rule 5: Gate Thresholds
**Canonical:** \(\tau_*\) (lowercase Greek tau with subscript)
**Examples:**
- \(\tau_\varepsilon\) — Residual threshold
- \(\tau_H\) — Energy threshold
- \(\tau_\ell\) — Soft gate threshold (generic)

**Rule:** Subscript indicates gate type

### Rule 6: Kuramoto Order Parameter
**Canonical:** \(R\) (coherence magnitude), \(\Phi\) (mean phase)
**Definition:** \(Z = \frac{1}{N}\sum_k e^{i\theta_k} = R e^{i\Phi}\)
**Rule:** Use \(R \in [0,1]\) for magnitude, \(\Phi \in [0, 2\pi)\) for phase

### Rule 7: Hash Chain
**Canonical:** \(\text{hash}(r_i) = h_i\), \(h_{i+1} = \text{hash}(r_{i+1} \| h_i)\)
**Rule:** Use \(\|\) for concatenation
**Alternative:** Use "parent_hash" field notation in code

### Rule 8: Layer Notation
**Canonical:** L0, L1, L2, L3, L4, L5 (uppercase L, digit subscript)
**Mapping:**
- L0 = Principle
- L1 = Mathematics
- L2 = Specification
- L3 = Runtime
- L4 = Operations
- L5 = Governance

**Rule:** Use uppercase \(L\) with numeric subscript; write "Layer L2" or "L2 document"

## Notation Migration Checklist

For documents using deprecated notation:

- [ ] Replace all \(\mathfrak C\) with \(C\)
- [ ] Replace all \(𝔉\) with \(C\)
- [ ] Verify Coherence/coherence capitalization
- [ ] Check acceptance set notation (\(\mathcal A\) vs plain \(A\))
- [ ] Verify UFE operator subscripts (\(L_{\text{phys}}\) vs \(L\))
- [ ] Check gate threshold notation (\(\tau_*\) vs \(T_*\))
- [ ] Update any layer notation inconsistencies
- [ ] Add notation reference link to document header
- [ ] Re-read for consistency

## Quick Reference Table

| Concept | Canonical | Deprecated | Context |
|---------|-----------|-----------|---------|
| Debt | \(C(x)\) | \(\mathfrak C(x)\), \(𝔉(x)\) | Residual aggregation |
| Residual norm | \(\tilde r_i\) | \(r_i^{\text{norm}}\) | Normalized component |
| Acceptance set | \(\mathcal A\) | \(A\) | Gate predicate domain |
| Physics evolution | \(L_{\text{phys}}\) | \(L\), \(\mathcal L\) | UFE component |
| Geom correction | \(S_{\text{geo}}\) | \(S\), \(\mathcal S\) | UFE component |
| Drive operators | \(G_i\) | \(\mathcal G_i\) | UFE component |
| Hard invariant | \(I_{\text{hard}}\) | \(I\), \(I_h\) | Gate type |
| Soft metric | \(q_\ell\) | \(q\) | Gate type |
| Energy | \(H(\Psi)\) | \(E(\Psi)\) | Hamiltonian functional |
| Order parameter | \(R\), \(\Phi\) | \(Z\), \(\rho\) | Phase coherence |

## References

- **Glossary:** [`coherence_spine/07_glossary/glossary.md`](../coherence_spine/07_glossary/glossary.md)
- **Debt Functionals:** [`coherence_math_spine/04_debt_functionals.md`](04_debt_functionals.md)
- **Specification Style Guide:** [`spec/10_lexicon.md`](../../spec/10_lexicon.md)
