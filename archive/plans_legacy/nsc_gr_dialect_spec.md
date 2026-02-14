# NSC_GR Dialect Specification

**Version:** 1.0  
**Date:** 2026-01-31  
**Status:** DRAFT

## Overview

The NSC_GR dialect extends the base Noetica Symbolic Compiler (NSC) with GR-specific primitives for PhaseLoom integration. This dialect enables direct specification of coherence gates, constraint enforcement, and spectral analysis in the NLLC language.

## Dialect Declaration

NLLC programs targeting GR/PhaseLoom should declare the dialect:

```nllc
# dialect: nsc_gr
# dialect_version: 1.0
```

## Extended Type System

### GR Tensor Types

```nllc
# Base tensor with metric signature
Tensor<T, D, signature>  # signature: (3,1) for 3+1 spacetime

# Specialized GR types
Metric           # Spatial metric γ_ij (symmetric 2-tensor)
ExtrinsicK       # Extrinsic curvature K_ij (symmetric 2-tensor)
Lapse            # Lapse function α (scalar)
Shift            # Shift vector β^i (vector)
Phi              # Conformal factor ψ (scalar)
GammaTilde       # Conformal connection Γ̃^i (vector)
Z4c              # Z4c constraint vector Z^i (vector)
```

### Constraint Types

```nllc
HamiltonianResidual   # H constraint residual (scalar)
MomentumResidual      # M^i constraint residual (vector)
ConstraintVector      # Combined constraint C_μ
```

### Coherence Types

```nllc
CoherenceGate         # Gate specification
CoherenceBudget       # Budget state
BandCoherence         # Per-band coherence Z_o
TailDanger           # Per-band danger D_o
OmegaActivity        # Spectral activity ω
```

## Extended Operators

### Geometric Operators

| Operator | Type Signature | Description |
|----------|---------------|-------------|
| `trace_sym6` | `Tensor<2> → Scalar` | Trace of symmetric 6-tensor |
| `inv_sym6` | `Tensor<2> → Tensor<2>` | Inverse of symmetric 2-tensor |
| `det_sym6` | `Tensor<2> → Scalar` | Determinant |
| `christoffel` | `Metric → Tensor<3>` | Christoffel symbols |
| `ricci_scalar` | `Metric, Tensor<3> → Scalar` | Ricci scalar |
| `ricci_tensor` | `Metric, Tensor<3> → Tensor<2>` | Ricci tensor |

### Constraint Operators

| Operator | Type Signature | Description |
|----------|---------------|-------------|
| `hamiltonian_constraint` | `Metric, ExtrinsicK, Scalar → HamiltonianResidual` | Compute H |
| `momentum_constraint` | `Metric, ExtrinsicK, Vector → MomentumResidual` | Compute M^i |
| `constraint_residual` | `ConstraintVector → Float` | L2 norm of constraints |

### Coherence Operators

| Operator | Type Signature | Description |
|----------|---------------|-------------|
| `gate_sem` | `CoherenceGate → Bool` | SEM barrier check |
| `gate_cons` | `CoherenceGate → Bool` | CONS barrier check |
| `gate_phy` | `CoherenceGate → Bool` | PHY barrier check |
| `compute_coherence` | `OmegaActivity → BandCoherence` | Z_o = \|⟨e^{iθ}⟩\| |
| `compute_danger` | `OmegaActivity → TailDanger` | D_o from variance |
| `band_octave` | `OmegaActivity, Int → OmegaActivity` | Dyadic band filter |

## Coherence Gate Specification

```nllc
# Declare coherence gates for a step
coherence_gate my_gate {
    sem: {
        eps_H_max = 1e-8,
        eps_M_max = 1e-8,
        energy_tolerance = 1e-10
    },
    cons: {
        hamiltonian_threshold = 1e-6,
        momentum_threshold = 1e-6,
        projection_margin = 0.01
    },
    phy: {
        cfl_number = 0.5,
        gauge_stability = 1e-8,
        singularity_margin = 1e-4
    }
}
```

## PhaseLoom Thread Specification

```nllc
# 27-thread lattice specification
phaseloom_config loom_27 {
    domains: [PHY, CONS, SEM],
    scales: [L, M, H],
    responses: [R0, R1, R2],
    
    # Thread-specific parameters
    thread_params: {
        PHY_L_R0: { dt_factor = 2.0 },
        PHY_M_R1: { dt_factor = 1.0, damping = 0.01 },
        CONS_H_R2: { dt_factor = 0.1, audit = true }
    }
}
```

## Spectral Analysis Primitives

```nllc
# FFT-based spectral computation
spectral_config omega_config {
    grid_size: [128, 128, 128],
    dx: 0.1,
    k_bins: [3, 3, 3],  # 27 spectral bins
    
    # Bin mapping functions
    kx_bin_map: kx -> floor((kx - kx_min) / delta_k),
    ky_bin_map: ky -> floor((ky - ky_min) / delta_k),
    kz_bin_map: kz -> floor((kz - kz_min) / delta_k)
}

# Compute omega (spectral activity)
omega = compute_omega(fields, spectral_config)
```

## Full Example: GR Step with Coherence

```nllc
# dialect: nsc_gr
# dialect_version: 1.0

# Input fields
metric gamma { type: Metric, shape: [128, 128, 128, 6] }
extrinsic_k K { type: ExtrinsicK, shape: [128, 128, 128, 6] }
lapse alpha { type: Lapse, shape: [128, 128, 128] }
shift beta { type: Shift, shape: [128, 128, 128, 3] }

# Output
metric gamma_out { type: Metric }
extrinsic_k K_out { type: ExtrinsicK }

# Spectral config
spectral_config spec { grid_size: [128, 128, 128], dx: 0.1 }

# Step with coherence gates
step gr_step(dt: Float, eps_H_max: Float, eps_M_max: Float) {
    # Compute RHS
    rhs = compute_rhs(gamma, K, alpha, beta)
    
    # Evolve fields
    gamma_out = gamma + dt * rhs.gamma
    K_out = K + dt * rhs.K
    
    # Compute constraints
    H = hamiltonian_constraint(gamma_out, K_out, alpha)
    M = momentum_constraint(gamma_out, K_out, beta)
    
    # Spectral coherence
    omega = compute_omega(gamma_out, spec)
    Z = compute_coherence(omega)
    D = compute_danger(omega)
    
    # Gate checks
    sem_pass = gate_sem(eps_H_max, eps_M_max, H, M)
    cons_pass = gate_cons(H, M)
    phy_pass = gate_phy(dt, alpha, beta)
    
    # Return with gate results
    return (gamma_out, K_out, H, M, Z, D, sem_pass, cons_pass, phy_pass)
}
```

## Integration with PhaseLoom Orchestrator

The NSC_GR dialect generates NIR that interfaces with PhaseLoom:

```nllc
# PhaseLoom integration directive
phaseloom_integration {
    receipt_format: "A:RCPT.step.accepted",
    invariants: [
        "N:INV.pde.div_free",
        "N:INV.pde.energy_nonincreasing",
        "N:INV.clock.stage_coherence"
    ],
    threads: ["A:THREAD.PHY.M.R1", "A:THREAD.CONS.M.R0"],
    omega_bands: 8
}
```

## Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Type extensions | ✅ DONE | GR types in nir.py |
| Geometric operators | ✅ DONE | intrinsic_binder.py |
| Constraint operators | ⚠️ PARTIAL | Basic forms exist |
| Coherence operators | ⚠️ PARTIAL | Need NIR lowering |
| PhaseLoom integration | ❌ PENDING | Not implemented |
| Spectral config | ⚠️ PARTIAL | In spectral/cache.py |

## References

- [`src/nllc/nir.py`](src/nllc/nir.py) - Extended NIR types
- [`src/nllc/intrinsic_binder.py`](src/nllc/intrinsic_binder.py) - Intrinsic bindings
- [`src/phaseloom/phaseloom_octaves.py`](src/phaseloom/phaseloom_octaves.py) - Band coherence
- [`src/core/gr_constraints.py`](src/core/gr_constraints.py) - Constraint computation
