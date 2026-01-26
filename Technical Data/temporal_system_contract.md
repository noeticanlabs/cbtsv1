# Temporal System Contract Specification

**Updated:** Reflects UnifiedClock architecture (`gr_solver/gr_clock.py`)

This contract is mandatory and defines what "time" means in the system. Violations result in SEM-hard failures.

## Entities

- **Physical time (t)**: The coordinate time in PDE evolution, representing the physical progression of the system state.
- **Audit/coherence time (τ)**: A monotone audit time that tracks coherence and verification, advancing only on accepted steps and remaining rollback-safe.
- **Indices (κ = (o, s, μ))**:
  - (o): Orchestration index, denoting the level in hierarchical solvers or multi-scale orchestration (e.g., global vs. local levels).
  - (s): Stage index, indicating the stage within a time-stepping scheme (e.g., Runge-Kutta stages).
  - (μ): Micro-step index, representing sub-steps or refinements within a stage (e.g., sub-iterations or micro-advances).
- **[`UnifiedClockState`](gr_solver/gr_clock.py:21)**: Single source of truth for all clock state. Shared across GRScheduler, MultiRateBandManager, and PhaseLoomMemory.

## UnifiedClock Integration

The temporal system now uses [`UnifiedClock`](gr_solver/gr_clock.py:119) as the central interface for time management:

```python
from gr_solver.gr_clock import UnifiedClock, UnifiedClockState

# Initialize unified clock
clock = UnifiedClock(base_dt=0.001, octaves=8)

# Access shared state
state = clock.get_state()  # UnifiedClockState instance

# Advance time
clock.tick(dt)

# For rollback: snapshot before step
snapshot = clock.snapshot()

# On rollback: restore state
clock.restore(snapshot)
```

## Hard Rules

### Accepted Step Semantics
Steps are accepted only if they satisfy all coherence constraints, residuals, and audit checks. Unaccepted steps trigger rollback without advancing physical time.

### Physical Time Semantics
Physical time (t) advances monotonically only on successfully accepted steps. Rollback reverts t to the last accepted state.

### Audit Time Semantics
Audit/coherence time (τ) advances monotonically on accepted steps, independent of physical time rollbacks. It serves as the immutable audit trail.

### Attempt Semantics
All computational attempts are logged, but they do not advance τ or contribute to accepted state until validated as coherent.

**Violation = SEM-hard**: Any breach of these rules constitutes a semantic hard error, halting the system and requiring manual intervention.