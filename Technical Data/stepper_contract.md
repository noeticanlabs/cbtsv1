# Stepper Contract Specification

## Overview
The stepper contract defines the mandatory interface and behavioral rules for the stepper component, which serves as the "court" that decides the evolution history in the temporal system. The stepper applies candidate time-steps to the state snapshot, validates coherence against the Law of Coherence (LoC), and either accepts or rejects the step. Acceptance advances physical time (t), audit time (τ), and immutable history; rejection triggers rollback with dt adjustment.

**Updated:** Reflects UnifiedClock architecture (`gr_solver/gr_clock.py`)

Violations of this contract constitute SEM-hard failures, halting the system and requiring manual intervention.

## Lexicon Declarations
This specification imports the following from the project lexicon (canon v1.2):
- **LoC_axiom**: Law of Coherence (primary invariant)
- **UFE_core**: Universal Field Equation evolution operator
- **CTL_time**: Control & time geometry layer
- **GR_dyn**: General Relativity dynamics

## Entities and Notations
- **X_n**: State snapshot at step n, encompassing all UFE state variables (Ψ in UFE notation)
- **t_n**: Physical time (coordinate time) at step n (from [`UnifiedClockState.global_time`](gr_solver/gr_clock.py:29))
- **τ_n**: Audit/coherence time at step n (monotone, rollback-safe)
- **Δt**: Candidate time-step size
- **rails_policy**: Constraint thresholds and gates (e.g., max ε_H, ε_M for Hamiltonian and momentum constraints)
- **phaseloom_caps**: Operational limits (e.g., max_attempts, dt_floor)
- **κ = (o, s, μ)**: Hierarchical indices for orchestration (o), stages (s), micro-steps (μ)
- **[`UnifiedClock`](gr_solver/gr_clock.py:119)**: Shared clock interface for time management

## Inputs
- `X_n`: Current state snapshot (UFE state Ψ_n)
- `t_n`: Current physical time
- `Δt`: Candidate time-step size
- `rails_policy`: Dictionary of constraint thresholds (e.g., `{'eps_H_max': 1e-4, 'eps_M_max': 1e-4}`)
- `phaseloom_caps`: Dictionary of caps (e.g., `{'max_attempts': 10, 'dt_floor': 1e-6}`)

## Outputs
- `accepted`: Boolean indicating acceptance
- If `accepted` is True:
  - `X_{n+1}`: Advanced state snapshot
  - `dt_used`: The Δt that was successfully used
  - `step_receipt`: Receipt object for the accepted step (advances τ)
- If `accepted` is False:
  - `X_n`: Rolled-back state (unchanged from input)
  - `dt_new`: Suggested new Δt for retry (e.g., halved Δt)
  - `rejection_reason`: String describing the violation type and details

## Behavioral Rules

### Acceptance Criteria
A step is accepted iff:
1. All coherence constraints are satisfied post-step (ε_H < rails_policy['eps_H_max'], ε_M < rails_policy['eps_M_max'])
2. No dt-dependent violations occur (CFL stability, stiffness checks)
3. No state-dependent violations occur (constraints not already violated pre-step or excessively worsened)
4. Audit checks pass (artifacts emitted, residuals logged)

### Violation Distinction
- **Dt-dependent violations**: Arise from numerical instability or scheme limitations
  - CFL violation: Δt > min(dx/c) or similar stability criterion
  - Stiffness violation: System eigenvalues indicate stiffness requiring implicit treatment or smaller Δt
  - Action: Reject, halve Δt (dt_new = Δt / 2)
- **State-dependent violations**: Arise from physical/incoherent state
  - Pre-step constraints violated: ε_H or ε_M already > thresholds at X_n
  - Excessive residual increase: Post-step residuals worsen beyond tolerance (e.g., > 10x pre-step)
  - Action: Reject, but may not adjust Δt if state is incoherent (flag for manual reset)

### Enforcement Mechanisms
- **Max Attempts**: If attempts ≥ phaseloom_caps['max_attempts'], fail hard with rejection_reason = "Max attempts exceeded"
- **Dt Floor**: If Δt ≤ phaseloom_caps['dt_floor'], fail hard with rejection_reason = "Dt floor reached"

### Emission Semantics
- **Attempt Receipt**: Emit for every attempt (accepted or rejected), logging residuals, dt used, violation checks. Does not advance τ.
- **Step Receipt**: Emit only on acceptance, advancing τ and contributing to immutable history.

### Prohibitions
- **No Continuation After Rejection**: Must not proceed to next step without changing Δt or state. Rejection requires rollback to X_n.
- **No Premature Step Labelling**: Must not label any computation as "step k" unless accepted and τ advanced.

## Failure Modes
- **Hard Fail**: Contract violation (e.g., advancing t without acceptance, missing receipts) → SEM-hard error.
- **Soft Fail**: Rejection with valid dt_new suggestion → Retry allowed within caps.

## Operational Meaning
The stepper enforces LoC by ensuring only coherent evolutions advance history. It integrates with PhaseLoom for multi-scale orchestration and provides rollback-safe audit trails via receipts.

**UnifiedClock Integration:** The stepper should use [`UnifiedClock`](gr_solver/gr_clock.py:119) for time management:
```python
from gr_solver.gr_clock import UnifiedClock

class GRStepper:
    def __init__(self, fields, rails_policy, phaseloom_caps, base_dt=0.001):
        self.clock = UnifiedClock(base_dt=base_dt)
        # ... other initialization
    
    def step(self, X_n):
        dt = self._compute_dt()
        # ... perform step
        self.clock.tick(dt)  # Advance unified clock
        return X_{n+1}
```

## Artifacts Generated
- Attempt receipts: Logged per attempt (residuals, gates checked)
- Step receipts: Immutable on acceptance (τ advance, Ω-ledger invariants)
- Rejection logs: Detailed violation reports for diagnostics

## Example Pseudocode
```python
from gr_solver.gr_clock import UnifiedClock

def stepper_court(X_n, clock: UnifiedClock, rails_policy, phaseloom_caps):
    """Stepper with UnifiedClock integration."""
    attempts = 0
    current_dt = clock.get_dt()
    
    while attempts < phaseloom_caps['max_attempts']:
        attempts += 1
        emit_attempt_receipt(X_n, clock.get_global_time(), current_dt, attempts)
        
        X_trial = apply_step(X_n, clock.get_global_time(), current_dt)  # RK4 or similar
        
        violations = check_violations(X_n, X_trial, current_dt, rails_policy)
        
        if not violations:
            emit_step_receipt(X_trial, clock.get_global_time() + current_dt, current_dt)
            clock.tick(current_dt)  # Advance unified clock
            return True, X_trial, current_dt, None
        else:
            violation_type, reason = classify_violation(violations)
            if violation_type == 'dt_dependent':
                current_dt = max(current_dt / 2, phaseloom_caps['dt_floor'])
            elif current_dt <= phaseloom_caps['dt_floor']:
                return False, X_n, None, "Dt floor reached: " + reason
            else:
                return False, X_n, current_dt, reason  # State-dependent, suggest same dt?
    
    return False, X_n, None, "Max attempts exceeded"
```

This contract ensures the temporal system remains coherent and auditable.