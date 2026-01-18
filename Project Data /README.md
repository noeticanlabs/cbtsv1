# LoC–GR/NR Technical Canvas Overview

This project implements the Law of Coherence (LoC) framework for General Relativity (GR) and Numerical Relativity (NR) simulations, providing persistent, explainable evolution through constraint damping, residual tracking, and multi-scale time governance.

## Core Equations

The fundamental evolution follows the UFE form:
```
∂_t Ψ = B(Ψ; θ) + λ K(Ψ; θ)
```

Where:
- **Ψ**: Full system state (metric fields γ_ij, extrinsic curvature K_ij, lapse α, shift β^i, etc.)
- **B**: Physics baseline (Einstein equations, gauge conditions)
- **K**: LoC coherence operator (damping + projection + stage coherence)
- **λ**: Coherence authority parameter
- **θ**: Parameters (damping coefficients κ_H, κ_M, etc.)

## LoC Augmentation

The LoC operator K_LoC decomposes as:
```
K_LoC = K_damp + K_proj + K_stage + K_bc
```

- **K_damp**: Constraint damping via ∇_Ψ C · K_damp = -κ_H ∇_H C - κ_M ∇_M C
- **K_proj**: Projection to manifold (e.g., det(γ̃) = 1 enforcement)
- **K_stage**: Stage coherence correction based on authoritative state tracking
- **K_bc**: Boundary coherence control

Residuals are computed as eps_H = max|H|, eps_M = ||M||_2, eps_proj = max|det(γ̃)-1|, eps_clk = max stage errors.

## Residuals

Residual tracking follows LoC-4 witness inequality:
```
||δ^{n+1}|| ≤ (1 - γ Δt_n) ||δ^n|| + ε^n + K η^n
```

Where ε^n bounds model residuals, η^n bounds representation defects, logged in Ω-ledger.

## PhaseLoom

PhaseLoom provides multi-thread coherence monitoring with 27 probe threads:
- Constraint threads (H, M components)
- Gauge threads (α, β^i deviations)
- Geometry threads (γ_ij, K_ij variations)

Computes phase θ = arctan2(b, a), amplitude ρ = √(a² + b²), rates ω = dθ/dt for braid coherence assessment.

## Omega-Ledger

Auditable hash-chain ledger with per-step receipts containing:
- Time stamps, step sizes, formulation/gauge IDs
- Residual norms (eps_H, eps_M, eps_proj, eps_clk)
- Gate states (G1-G4: constraint bounds, clock coherence, projection, damage budget)
- Coherence hashes (pre/post/step/chain)
- LoC-PRINCIPLE fields: delta_norm, eps_model, eta_rep, gamma, clock_tau, clock_margin

Chain integrity verified via SHA256(serialized_data + prev_hash).

## Gates

Coherence gates enforce safety:
- **G1**: eps_H ≤ H_max ∧ eps_M ≤ M_max
- **G2**: eps_clk ≤ clk_max
- **G3**: eps_proj ≤ proj_max
- **G4**: D ≤ D_prev + budget (damage monotonicity)

Failure triggers correction (dt shrink, gain adjustment) or rollback.

## Tests

Comprehensive test suite covers:
- Initialization (Minkowski metric, finite fields, det(γ) > 0)
- Memory operations (AeonicMemoryBank put/get/maintenance)
- Single/multi-step evolution (finite fields, dt arbitration)
- Rails/gates functionality (margin computation, violation detection)
- PhaseLoom threads (residual computation, dt arbitration)
- Constraints/geometry (finite Christoffels/Ricci, bounded residuals)

## Integration of LoC-PRINCIPLE and LoC-Time

LoC-PRINCIPLE provides theorem-generating framework with lemmas LoC-1 through LoC-6 for constraint preservation, damped recovery, witness bounds, clock coherence, and representation fidelity.

LoC-Time implements multi-clock governance (Levels 1-5) with PhaseLoom braid metrics, extreme-λ dominance safeguards, and deterministic rollback/rails.

Integration ensures persistent evolution via damped coherence (LoC-3/4), clock respect (LoC-5), and auditable residuals (Ω-ledger with LoC-6 eta_rep).