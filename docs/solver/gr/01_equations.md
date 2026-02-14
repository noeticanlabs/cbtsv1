# Equations Implemented

## 3+1 Decomposition

The solver uses the 3+1 (Arnowitt-Deser-Misner) decomposition of Einstein's equations:

```
ds² = -α²dt² + γᵢⱼ(dxⁱ + βⁱdt)(dxʲ + βʲdt)
```

Where:
- α is the lapse function
- βⁱ is the shift vector
- γᵢⱼ is the 3-metric

## Evolution Equations

### BSSN Formulation

The solver implements the BSSN (Baumgarte-Shapiro-Shibata-Nakamura) formulation:

1. **Conformal factor**: γ = det(γᵢⱼ), γ̃ᵢⱼ = e^{-4φ}γᵢⱼ
2. **Trace-free extrinsic curvature**: Āᵢⱼ = Kᵢⱼ - (1/3)γᵢⱼK
3. **Conformal connection**: Γ̃ⁱ = γ̃^{ⱼk}Γ̃ⁱ_{ⱼk}

### Hamiltonian Constraint

```
H = R + K² - KᵢⱼKᵢⱼ = 0
```

### Momentum Constraints

```
Mⁱ = Dⱼ(Kᵢⱼ - γᵢⱼK) = 0
```

## RHS Computations

See [`gr_rhs.py`](../../src/cbtsv1/solvers/gr/equations/rhs.py) for the implementation of the right-hand side equations.
