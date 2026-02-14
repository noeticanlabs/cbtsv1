# State and Variables

## State Vector

The GR solver maintains the following state variables:

### BSSN Variables

| Variable | Symbol | Description |
|----------|--------|-------------|
| Conformal metric | γ̃ᵢⱼ | Conformal 3-metric |
| Extrinsic curvature | Āᵢⱼ | Trace-free extrinsic curvature |
| Conformal factor | φ | log(det(γ)/det(γ₀)) / 12 |
| Trace K | K | Trace of extrinsic curvature |
| Conformal connection | Γ̃ⁱ | Conformal connection functions |

### Gauge Variables

| Variable | Symbol | Description |
|----------|--------|-------------|
| Lapse | α | α = -nᵢnⁱ |
| Shift | βⁱ | Shift vector |

## Core Fields

See [`gr_core_fields.py`](../../src/cbtsv1/solvers/gr/geometry/core_fields.py) for the implementation.

## Initialization

The state can be initialized for:
- Minkowski (flat) spacetime
- Schwarzschild (isotropic or Kerr-Schild coordinates)
- Custom initial data
