# Solver Contract Specification

## Overview
The solver contract defines the mandatory interface and behavioral rules for the solver component, which computes the right-hand side (RHS) of the evolution equations in the UFE framework. The solver handles multi-stage evaluations for Runge-Kutta methods or similar, ensuring all computations use correct stage times to prevent staging bugs (e.g., MMS violations). It manages gauge policies and time-dependent sources, returning SEM failures if prerequisites are unmet.

Violations of this contract constitute SEM-hard failures, halting the system.

## Lexicon Declarations
This specification imports the following from the project lexicon (canon v1.2):
- **LoC_axiom**: Law of Coherence (primary invariant)
- **UFE_core**: Universal Field Equation evolution operator
- **CTL_time**: Control & time geometry layer
- **GR_dyn**: General Relativity dynamics

## Entities and Notations
- **X**: State vector (UFE state Ψ)
- **t**: Physical time (coordinate time)
- **gauge_policy**: Dictionary specifying gauge choices (e.g., `{'gauge_type': 'harmonic', 'params': {...}}`)
- **S(t)**: Sources function, potentially time-dependent
- **RHS**: Right-hand side F(X,t) + S(t)
- **t^{(μ)}**: Stage time for stage μ, where t^{(μ)} = t_n + c_μ Δt (c_μ are Runge-Kutta coefficients)
- **Diagnostics**: Per-block diagnostics, e.g., A/B/C/D for Gammã (Christoffel symbols) blocks

## Inputs
- `X`: State vector
- `t`: Time (stage time t^{(μ)} for stage evaluations)
- `gauge_policy`: Gauge policy dictionary
- `S(t)`: Sources function, accepts time `t` explicitly if time-dependent

## Outputs
- `RHS`: Computed F(X,t) + S(t) at the specified stage time `t`
- Optionally: `diagnostics`: Dictionary of per-block diagnostics (e.g., `{'Gamma_tilde_A': ..., 'Gamma_tilde_B': ..., 'Gamma_tilde_C': ..., 'Gamma_tilde_D': ...}`)

## Behavioral Rules

### Hard Rules
1. All stage evaluations MUST use stage time (t^{(μ)}=t_n+c_μ Δt). Using base time t_n constitutes a contract violation.
2. If sources S are declared time-dependent, the solver MUST accept time `t` explicitly as an argument.
3. If prerequisites for RHS computation are not initialized (e.g., Christoffels, gauge derivatives), the solver MUST return a SEM failure, not zero-filled placeholders.

### Additional Rules
- The solver MUST enforce gauge policy consistency across stages.
- Diagnostics MUST be computed per-block if requested, providing granular insight into RHS components.

## Failure Modes
- **SEM-hard failure**: Prerequisites not initialized (e.g., missing Christoffels) → Return failure code with detailed reason, do not proceed.
- **Staging violation**: Incorrect time usage → SEM-hard error.

## Operational Meaning
The solver ensures accurate RHS computation for coherent evolution, preventing numerical artifacts from staging errors. It integrates with the stepper for multi-stage methods and emits diagnostics for debugging GR dynamics.

## Artifacts Generated
- RHS vectors at stage times
- Optional diagnostic logs per block
- Failure receipts if prerequisites fail

## Example Pseudocode
```python
def compute_rhs(X, t, gauge_policy, S_func):
    # Check prerequisites
    if not prerequisites_initialized(X, gauge_policy):
        return SEM_FAILURE, "Prerequisites not initialized: Christoffels missing"
    
    # Compute F(X,t)
    F = compute_F(X, t, gauge_policy)
    
    # Compute S(t) - pass t explicitly if time-dependent
    S = S_func(t)
    
    rhs = F + S
    
    # Optional diagnostics
    diags = compute_diagnostics(X, t, gauge_policy)  # e.g., Gamma_tilde blocks
    
    return rhs, diags
```

This contract ensures staging correctness and prerequisite integrity in the solver.