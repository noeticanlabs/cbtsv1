# Constraints

## Hamiltonian Constraint

The Hamiltonian constraint enforces energy conservation:

```
H = R + K² - KᵢⱼKᵢⱼ = 0
```

Where:
- R is the 3-dimensional Ricci scalar
- K is the trace of extrinsic curvature
- Kᵢⱼ is the extrinsic curvature tensor

## Momentum Constraints

The momentum constraints enforce momentum conservation:

```
Mⁱ = Dⱼ(Kᵢⱼ - γᵢⱼK) = 0
```

Where Dⱼ is the covariant derivative associated with γᵢⱼ.

## Constraint Monitoring

The solver computes constraint violations at each timestep. See:

- [`gr_constraints.py`](../../src/cbtsv1/solvers/gr/constraints/constraints.py) - Constraint evaluation
- [`hard_invariants.py`](../../src/cbtsv1/solvers/gr/hard_invariants.py) - Hard invariant checks

## Constraint Damping

The solver implements constraint-damping terms in the BSSN formulation to suppress constraint violations.
