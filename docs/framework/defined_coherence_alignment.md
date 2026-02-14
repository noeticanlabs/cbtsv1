# Defined Coherence Alignment

This document describes how cbtsv1 aligns with the **Defined Coherence** canon from `coherence-framework`.

## Canonical Definition

The coherence_functional is defined as:

$$\mathfrak c(x) = \langle \tilde r(x), W \tilde r(x) \rangle, \quad \tilde r = S^{-1} r$$

Where:
- **r**: Vector of constraint residuals
- **S**: Scaling matrix (per-block diagonal entries)
- **W**: Weight matrix (diagonal covariance)
- **r̃**: Scaled residual vector

## cbtsv1 Residual Blocks

For GR evolution, cbtsv1 produces two primary residual blocks:

| Block | Physics | Vector Shape | Description |
|-------|---------|--------------|-------------|
| `hamiltonian` | $H = R + K^2 - K_{ij}K^{ij} - 2\Lambda$ | $(N_x, N_y, N_z)$ | Hamiltonian constraint (scalar) |
| `momentum` | $M^i = D_j(K^{ij} - \gamma^{ij}K)$ | $(N_x, N_y, N_z, 3)$ | Momentum constraint (vector) |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      cbtsv1 Solver                          │
│  ┌─────────────────┐    ┌────────────────────────────────┐  │
│  │ GRConstraints  │───▶│ defined_coherence_blocks.py   │  │
│  │ (H, M computed)│    │ build_residual_blocks()       │  │
│  └─────────────────┘    └──────────────┬─────────────────┘  │
│                                        │                    │
│                                        ▼                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │      coherence_integration.py                        │   │
│  │      compute_gr_coherence()                          │   │
│  │      ┌────────────────────────────────────────┐     │   │
│  │      │ vendor/coherence_framework/core.py     │     │   │
│  │      │ compute_coherence(blocks)              │     │   │
│  │      │ 𝔠 = Σ weight_i × ||scale_i × r_i||² │     │   │
│  │      └────────────────────────────────────────┘     │   │
│  └──────────────────────────┬───────────────────────────┘   │
│                             │                               │
│                             ▼                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ GRLedger / Receipts                                 │   │
│  │ - coherence_value: float                            │   │
│  │ - residual_blocks: {dim, l2, linf, hash}           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Integration Point

Canonical coherence is computed in the stepper after constraint residuals are computed:

```python
from cbtsv1.solvers.gr.coherence_integration import compute_gr_coherence

# After constraints are computed (H, M)
coherence_result = compute_gr_coherence(
    fields, 
    constraints, 
    config
)

# Log to ledger
ledgers['coherence_value'] = coherence_result['coherence_value']
ledgers['residual_blocks'] = coherence_result['blocks']
```

## Scale/Weight Configuration

Configuration is stored in `config/defined_coherence_gr.json`:

```json
{
  "version": "1.0.0",
  "covariance_model": "diag",
  "blocks": {
    "hamiltonian": {
      "scale": 1.0,
      "weight": 1.0
    },
    "momentum": {
      "scale": 1.0,
      "weight": 1.0
    }
  }
}
```

- **scale**: Applied to residual vector: $\tilde r = scale \times r$
- **weight**: Multiplies L2 squared: $contribution = weight \times ||\tilde r||^2$

## Receipt Schema

Receipts include canonical coherence data:

```json
{
  "step": 42,
  "coherence_value": 1.234567e-8,
  "residual_blocks": {
    "hamiltonian": {
      "dim": 512,
      "l2": 3.456789e-5,
      "linf": 1.234567e-4,
      "hash": "a1b2c3d4...",
      "scale": 1.0,
      "weight": 1.0
    },
    "momentum": {
      "dim": 1536,
      "l2": 2.345678e-5,
      "linf": 9.876543e-5,
      "hash": "e5f6g7h8...",
      "scale": 1.0,
      "weight": 1.0
    }
  }
}
```

## Testing

Three alignment tests verify canonical behavior:

1. **Core coherence test**: Verify `compute_coherence()` returns expected values
2. **Minkowski test**: Verify coherence ≈ 0 for flat spacetime
3. **Artificial residual test**: Verify coherence matches hand calculation

Run tests:
```bash
pytest tests/test_coherence_alignment.py -v
```

## What Was Changed

| Old (Deprecated) | New (Canonical) |
|-------------------|----------------|
| `gr_coherence.py` | Use `coherence_integration.py` |
| `eps_H`, `eps_M` norms | `coherence_value` + block summaries |
| Solver-local coherence math | Canonical `compute_coherence()` |

## Reference

- Canonical definition: [coherence-framework/docs/coherence/01_canonical_definition.md](https://github.com/coherence-framework/docs)
- This implementation: `src/cbtsv1/solvers/gr/coherence_integration.py`
