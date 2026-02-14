# Conformance and Validation

## Validation Steps

1. **Constraint Check**: Verify H < tolerance, M < tolerance
2. **Energy Conservation**: Check Hamiltonian constraint
3. **Momentum Conservation**: Check momentum constraints
4. **Bianchi Identities**: Verify geometric identities

## Tolerances

| Test | Target Tolerance |
|------|-----------------|
| Minkowski | 10⁻¹² |
| Schwarzschild | 10⁻⁶ |
| Convergence | Order N |

## Running Conformance

```python
from cbtsv1.solvers.gr.hard_invariants import check_invariants

# Check constraints
is_conformant = check_invariants(state, tolerance=1e-6)
```

## Step Rejection

If constraints exceed tolerance, the step is rejected and retried with smaller timestep.
