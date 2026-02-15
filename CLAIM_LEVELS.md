# Claim Levels for CBTSV1 Physics Modules

This document defines the claim level convention for physics-facing modules in CBTSV1.

## Claim Level Definitions

| Level | Name | Description | Requirements |
|-------|------|-------------|--------------|
| 1 | Engineering Evidence | Bounded stability, empirical validation | Test results, numerical experiments |
| 2 | Mathematical Backing | Coercivity, spectral-type statements | Formal proofs or derivations |
| 3 | Theorem Claims | Field-level theorem (requires proof artifacts) | Peer-reviewed proof, proof assistants |

## Usage

Add claim level to module headers:

```python
# CLAIM_LEVEL = 1  # Engineering evidence only
# CLAIM_LEVEL = 2  # Mathematical backing available  
# CLAIM_LEVEL = 3  # Theorem-level claim

CLAIM_LEVEL = 1
```

## Module Claim Levels

| Module | Level | Notes |
|--------|-------|-------|
| `cbtsv1.solvers.gr.geometry` | 1 | Empirical validation |
| `cbtsv1.solvers.gr.constraints` | 1 | Numerical experiments |
| `cbtsv1.solvers.gr.gauge` | 1 | Engineering evidence |
| `cbtsv1.solvers.gr.phases` | 1 | Phase logic testing |
| `cbtsv1.solvers.gr.coherence_integration` | 1 | Defined coherence alignment |

## Test Requirements

Tests should declare which claim level they exercise:

```python
# @pytest.mark.claim_level(1)  # Engineering evidence test
# @pytest.mark.claim_level(2)  # Mathematical validation test
# @pytest.mark.claim_level(3)  # Theorem proof test

def test_gr_stability():
    """Claim level 1: Engineering evidence for bounded stability."""
    ...
```

## Governance

- **Level 1 claims**: Can be made by any contributor with test evidence
- **Level 2 claims**: Requires mathematical derivation or formal analysis
- **Level 3 claims**: Requires external review and proof artifacts

This convention helps prevent reputational damage from overclaiming and clarifies the maturity of each component.
