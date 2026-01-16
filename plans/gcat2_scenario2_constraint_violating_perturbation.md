# GCAT-2 Scenario 2: Constraint-Violating Initial Perturbation
## Overview
This scenario initializes the spacetime with small but nonzero Hamiltonian (H) and Momentum (M_i) constraints, while setting extrinsic curvature (K_ij) and spatial metric (γ_ij) appropriately. It tests whether active constraint damping fields (Z, Z_i) can suppress violations without leading to numerical explosion.

## Emergent Nature
Constraint violations often emerge in numerical GR from truncation errors in initial data preparation or evolution. Even with constraint-preserving schemes, small residuals can accumulate, especially in dynamic regimes. This scenario models such natural emergence by introducing controlled violations that evolve through the system's dynamics, testing the robustness of constraint damping mechanisms.

## Stress on the Three Monsters
- **Geometry Monster**: Constraint violations can manifest as artificial singularities or metric discontinuities, destabilizing geometric evolution.
- **Constraint Monster**: Directly tests the ability to maintain H ≈ 0 and M_i ≈ 0 through active damping, exposing weaknesses in Z-field enforcement.
- **Spectral Monster**: Violations can excite high-frequency modes in constraint fields, challenging spectral resolution and filtering.

## Setup
### Initial Conditions
- Use Bowen-York or similar initial data for black hole or binary system.
- Introduce perturbations: H = ε * random_field, M_i = ε * grad(random_field), where ε = 10^-4.
- Set γ_ij to satisfy Einstein equations perturbatively, e.g., γ_ij = γ_0_ij + δγ_ij to compensate for K_ij adjustments.
- Ensure Z, Z_i are initialized with small values to activate damping.

### Triggers
- Evolve for 50-100 time units, monitoring constraint norms without external intervention.

## Measurable Outcomes
### Success Criteria
- Constraint damping: H/M norms decrease monotonically by >50% over first 20 time units.
- No explosion: All field values remain bounded (<10^6 in normalized units).
- Geometric stability: Ricci scalar fluctuations <10^-2.
- Spectral convergence: High-k modes in constraints damped without aliasing artifacts.

### Failure Criteria
- Explosion: Any field exceeds 10^10, indicating instability.
- Non-damping: H/M norms increase by >20% after initial period.
- Geometric breakdown: Metric becomes non-positive definite.
- Spectral failure: Constraint oscillations persist with amplitude >ε.

## Integration into Test Framework
- Implement as `test_gcat2_scenario2()` in `tests/test_gcat2_emergent_failures.py`.
- Utilize `gr_constraints.py` for norm computation and `gr_ledger` for history tracking.
- Integrate with phaseloom to ensure damping fields are active.
- Test passes if success criteria met; logs violations for analysis.
- Receipts: Store constraint evolution data in JSON for post-processing.