# Gauge Conditions

## Lapse Conditions

The solver supports multiple gauge choices:

### 1. Harmonic Gauge
```
∂ₜα = -α² K
```

### 2. 1+Log Slicing
```
∂ₜα = α (2K - ∂ₜln(γ))
```

### 3. Maximal Slicing
```
K = 0 (trace of extrinsic curvature = 0)
```

## Shift Conditions

### 1. Minimal Distortion
```
∂ₜβⁱ = βⁱ ∂ⱼβⱼ - γⁱⱼ∂ⱼ(ln(α/γ¹ᐟ²))
```

### 2. Gamma-Driver
```
∂ₜβⁱ = ξ(Γ̃ⁱ - Γ̃ⁱ₀) - ηβⁱ
```

See [`gr_gauge.py`](../../src/cbtsv1/solvers/gr/gauge/gauge.py) for the implementation.

## Gauge Evolution

The gauge is evolved to:
- Avoid singularities (singularity avoidance)
- Maintain well-posedness
- Optimize computational efficiency
