# PhaseLoom Contract Specification

## Overview

The PhaseLoom contract defines the mandatory interface and behavioral rules for the PhaseLoom system, a 27-thread invariant lattice for phase-space control in the CBTSV1 project. PhaseLoom implements the "Sensor-Governor" lattice architecture defined in Aeonic PhaseLoom Canon v1.0, providing multi-scale, multi-domain oversight of the numerical evolution. It coordinates with the solver via the GR adapter, the stepper via the gate system, and the orchestrator via memory management.

PhaseLoom operates as a hierarchical lattice where 27 threads (3 domains × 3 scales × 3 response tiers) monitor and control the numerical simulation. Each thread tracks residuals in its domain/scale/response sector and provides corrective actions (rails) when thresholds are violated. The system integrates with octaves for frequency-band analysis and provides a comprehensive memory model for state tracking.

Violations of this contract constitute SEM-hard failures, halting the system and requiring manual intervention.

## Lexicon Declarations

This specification imports the following from the project lexicon (canon v1.2):
- **LoC_axiom**: Law of Coherence (primary invariant)
- **UFE_core**: Universal Field Equation evolution operator
- **CTL_time**: Control & time geometry layer
- **GR_dyn**: General Relativity dynamics
- **Aeonic_Phaseloom**: PhaseLoom system specification

## Architecture Overview

### 27-Thread Invariant Lattice

PhaseLoom employs a 3×3×3 lattice structure that partitions the monitoring space:

```
Domains (D):     PHY (Physics), CONS (Constraints), SEM (Semantics)
Scales (S):      L (Large), M (Medium), H (High/Tail)
Responses (R):   R0 (FAST - Immediate), R1 (MID - Stabilizing), R2 (SLOW - Governance)
```

The 27 threads are identified by tuples (D, S, R) and can be referenced as strings (e.g., "PHY_L_R0", "CONS_M_R1", "SEM_H_R2").

**Thread Structure:**
- [`ThreadState`](src/phaseloom/phaseloom_27.py:5): Per-thread state container
- `domain`: Primary domain classification
- `scale`: Temporal/spatial scale classification
- `response`: Response tier classification
- `residual`: Current residual value for this thread
- `dt_cap`: Maximum allowable timestep from this thread
- `active`: Whether this thread participates in arbitration
- `action_suggestion`: Suggested corrective action (optional)

### Thread Categories by Domain

**PHY (Physics) Threads:**
- Monitor numerical discretization errors and CFL conditions
- R0 (FAST): Immediate timestep constraints from high-frequency modes
- R1 (MID): Stability and convergence monitoring
- R2 (SLOW): Governance and audit-triggering

**CONS (Constraints) Threads:**
- Monitor constraint violations (Hamiltonian and momentum constraints)
- L (Large): Macro-scale drift detection
- M (Medium): Mid-scale boundary and reflection issues
- H (High): Micro-scale spike detection

**SEM (Semantic) Threads:**
- Monitor semantic invariants and policy compliance
- R0: Immediate semantic violations
- R1: Stabilizing semantic checks
- R2: Governance and halt conditions

## Core Components

### PhaseLoom27 ([`src/phaseloom/phaseloom_27.py`](src/phaseloom/phaseloom_27.py:14))

The central 27-thread lattice controller.

**Key Methods:**
- [`__init__()`](src/phaseloom/phaseloom_27.py:40): Initialize the lattice with 27 threads
- [`update_residual(domain, scale, value)`](src/phaseloom/phaseloom_27.py:52): Update residual for a domain/scale pair, propagating to all response tiers
- [`arbitrate_dt()`](src/phaseloom/phaseloom_27.py:80): Find minimum dt cap across all threads and identify dominant thread
- [`check_gate_step(thresholds)`](src/phaseloom/phaseloom_27.py:104): Enforce LoC inequalities for Gate_step
- [`check_gate_orch(window_stats, thresholds)`](src/phaseloom/phaseloom_27.py:140): Enforce LoC inequalities for Gate_orch
- [`get_rails(dominant_thread)`](src/phaseloom/phaseloom_27.py:171): Get pre-authorized corrective actions

**Default Thresholds:**
```python
DEFAULT_THRESHOLDS = {
    'SEM': 0.0,    # Hard semantic barrier
    'CONS': 1e-6,  # High-fidelity constraint tolerance
    'PHY': 1e-4    # Evolution discretization tolerance
}
```

### PhaseLoomOctaves ([`src/phaseloom/phaseloom_octaves.py`](src/phaseloom/phaseloom_octaves.py:16))

Frequency-band analysis for multi-rate temporal decomposition.

**Key Methods:**
- [`add_omega_sample(omega_current)`](src/phaseloom/phaseloom_octaves.py:26): Add current omega sample to history
- [`compute_dyadic_bands()`](src/phaseloom/phaseloom_octaves.py:33): Compute dyadic moving-average differences
- [`compute_band_coherence(omega_band)`](src/phaseloom/phaseloom_octaves.py:61): Compute coherence C_o for each band
- [`compute_tail_danger(omega_band)`](src/phaseloom/phaseloom_octaves.py:73): Compute tail danger D_o
- [`process_sample(omega_current)`](src/phaseloom/phaseloom_octaves.py:88): Full processing pipeline

**Lexicon Symbols:**
- $\\omega^{(o)}$: Aeonic_Phaseloom.octave_rate
- $C_o$: Aeonic_Phaseloom.band_coherence
- $D_o$: Aeonic_Phaseloom.tail_danger

### PhaseLoomMemory ([`src/phaseloom/phaseloom_memory.py`](src/phaseloom/phaseloom_memory.py:7))

Memory and state tracking for PhaseLoom operations.

**Key Methods:**
- [`should_compute_loom(...)`](src/phaseloom/phaseloom_memory.py:77): Determine if loom computation is needed
- [`post_loom_update(loom_data, step)`](src/phaseloom/phaseloom_memory.py:217): Update state after loom computation
- [`get_bands_to_update()`](src/phaseloom/phaseloom_memory.py:259): Get mask of bands updated
- [`check_regime_shift()`](src/phaseloom/phaseloom_memory.py:302): Check if regime shift detected

**Memory Tiers:**
- `step_count`: Global step counter
- `last_loom_step`: Last step where loom was computed
- `prev_K`, `prev_gamma`: Previous state for delta computation
- `dominant_band`: Current dominant frequency band
- `amplitude`: Current signal amplitude
- `bands_updated`: Mask of bands updated in last computation

### PhaseLoomGRAdapter ([`src/phaseloom/phaseloom_gr_adapter.py`](src/phaseloom/phaseloom_gr_adapter.py:1))

Adapter for GR solver integration. Translates between solver outputs and PhaseLoom inputs.

### PhaseLoomGRController ([`src/phaseloom/phaseloom_gr_controller.py`](src/phaseloom/phaseloom_gr_controller.py:1))

Central controller for GR-specific PhaseLoom operations.

### PhaseLoomGROrchestrator ([`src/phaseloom/phaseloom_gr_orchestrator.py`](src/phaseloom/phaseloom_gr_orchestrator.py:1))

Orchestrator integration for PhaseLoom.

### PhaseLoomRailsGR ([`src/phaseloom/phaseloom_rails_gr.py`](src/phaseloom/phaseloom_rails_gr.py:1))

Rails (corrective actions) implementation for GR system.

### PhaseLoomReceiptsGR ([`src/phaseloom/phaseloom_receipts_gr.py`](src/phaseloom/phaseloom_receipts_gr.py:1))

Receipt generation and management for PhaseLoom operations.

### PhaseLoomRenderGR ([`src/phaseloom/phaseloom_render_gr.py`](src/phaseloom/phaseloom_render_gr.py:1))

Rendering and visualization support for PhaseLoom diagnostics.

### PhaseLoomThreadsGR ([`src/phaseloom/phaseloom_threads_gr.py`](src/phaseloom/phaseloom_threads_gr.py:1))

Thread management utilities for GR system.

## Contract Interfaces

### Solver Interface (via GR Adapter)

PhaseLoom receives solver residuals and outputs:

**Inputs from Solver:**
- `residual`: Per-domain residual values from RHS computation
- `stage_time`: Current stage time t^(μ)
- `gauge_policy`: Current gauge configuration

**Outputs to Solver:**
- `dt_cap`: Maximum allowable timestep
- `rails`: Pre-authorized corrective actions

**Behavioral Rules:**
1. Solver MUST provide residuals organized by domain (PHY, CONS, SEM)
2. Adapter MUST normalize residuals before updating threads
3. PhaseLoom MUST return dt_cap before next stage computation

### Stepper Interface (via Gate System)

PhaseLoom provides gate validation for step acceptance:

**Inputs from Stepper:**
- `candidate_state`: State snapshot for validation
- `dt_candidate`: Proposed timestep
- `window_stats`: Window statistics for Gate_orch

**Outputs to Stepper:**
- `gate_passed`: Boolean for step acceptance
- `violation_reasons`: List of violation descriptions
- `rails`: Corrective actions if violations detected

**Gate Types:**
- **Gate_step**: Per-step coherence validation (3 barriers: SEM, CONS, PHY)
- **Gate_orch**: Window-level regime stability (chatter, residual thresholds)

### Orchestrator Interface (via Memory)

PhaseLoom integrates with orchestrator for regime decisions:

**Inputs from Orchestrator:**
- `window_definition`: Aggregation window (num_steps or delta_tau)
- `performance_counters`: Performance metrics

**Outputs to Orchestrator:**
- `dominant_thread`: Current controlling thread
- `window_receipt`: Window-level receipt
- `regime_label`: Current regime classification

## Memory Model

### PhaseLoomMemory Integration

PhaseLoomMemory provides state tracking across operations:

**Clock System Support:**
- [`UnifiedClock`](src/core/gr_clock.py:119): Shared time state interface
- [`MultiRateClockSystem`](src/core/gr_clocks.py:1): Legacy multi-rate clock (backward compatibility)

**Band Tracking:**
- 8 octaves (0-7) for frequency decomposition
- `bands_updated`: Boolean mask indicating which bands were updated
- `dominant_band`: Index of most energetic band
- `amplitude`: Signal amplitude for culling decisions

**Octave Culling:**
Non-dominant bands can be culled if:
```python
threshold = octave_cull_threshold * (2 ** (dominant_band - o))
if delta_K < threshold and delta_gamma < threshold:
    bands_to_update[o] = False
```

### Regime Hash Computation

PhaseLoomMemory computes regime hashes for caching:
- Combines `base_dt`, `dominant_band`, `resolution`
- Used to invalidate caches when regime shifts
- Stored in clock system for diagnostics

## Receipt Integration

### Receipt Types

PhaseLoom generates receipts at multiple levels:

**Attempt Receipts:**
- Emitted for every loom computation attempt
- Contains residuals, rails checked, dominance metrics
- Does not advance audit time τ

**Step Receipts:**
- Emitted when loom computation completes
- Contains band metrics, coherence values, danger levels
- Advances τ and contributes to immutable history

**Window Receipts:**
- Aggregated over canonical windows
- Contains mean residuals, regime statistics
- Generated via orchestrator integration

### Receipt Data Structure

```python
{
    'step': int,
    'dominant_thread': Tuple[str, str, str],
    'dt_cap': float,
    'rails': List[Dict],
    'omega_band': np.ndarray,
    'C_band': np.ndarray,  # Coherence per band
    'D_band': np.ndarray,  # Danger per band
    'D_max': float,
    'C_global': float,
    'dominant_band': int,
    'amplitude': float
}
```

## Thread Coordination

### Arbitration Process

When multiple threads impose dt caps, PhaseLoom selects the minimum:

```python
def arbitrate_dt(self) -> Tuple[float, Tuple[str, str, str]]:
    min_dt = float('inf')
    dominant_key = None
    
    for key, thread in self.threads.items():
        if not thread.active:
            continue
        if thread.dt_cap < min_dt:
            min_dt = thread.dt_cap
            dominant_key = key
    
    return min_dt, dominant_key
```

### Rail Selection

Rails are selected based on the dominant thread:

**PHY Rails:**
- PHY.H dominance: Increase dissipation (strength=1.5)
- PHY.R2 dominance: Trigger full audit

**CONS Rails:**
- CONS.L dominance: Enforce projection (kreiss-oliger)
- CONS.M dominance: Adjust boundary (absorb mode)
- CONS.H dominance: Increase dissipation (strength=2.0)

**SEM Rails:**
- SEM.R2 dominance: Halt and dump
- Other SEM dominance: Log warning

### Gate Step Validation

```python
def check_gate_step(self, thresholds=None):
    thresholds = thresholds or DEFAULT_THRESHOLDS
    
    # Aggregate max residuals per domain
    max_residuals = {d: 0.0 for d in self.DOMAINS}
    for (d, s, r), thread in self.threads.items():
        if thread.active:
            max_residuals[d] = max(max_residuals[d], thread.residual)
    
    # Check barriers
    for domain in self.DOMAINS:
        val = max_residuals[domain]
        limit = thresholds.get(domain, float('inf'))
        if val > limit:
            return False, [f"{domain} violation: {val:.2e} > {limit:.2e}"]
    
    return True, []
```

## Error Handling

### Failure Modes

**SEM-hard Failures:**
- Missing thread initialization
- Invalid domain/scale/response indices
- Gate step violation without proper rejection
- Receipt generation failure

**Soft Failures:**
- Rejection with valid dt_new suggestion
- Regime mislabel (log and retry)
- Octave culling optimization failure

### Recovery Procedures

**Thread State Reset:**
If a thread becomes tainted or invalid:
1. Set `active = False` for the thread
2. Emit warning receipt
3. Re-arbitrate dt without the thread
4. If all threads tainted, trigger full reset

**Rollback Integration:**
When rollback occurs:
1. PhaseLoomMemory sets `tainted = True`
2. Force loom computation on next opportunity
3. Re-validate all thread states
4. Emit recovery receipt

### Violation Classification

**Dt-Dependent Violations:**
- Arise from numerical instability
- Action: Reject, halve dt
- Examples: CFL violation, stiffness violation

**State-Dependent Violations:**
- Arise from physical/incoherent state
- Action: Reject, may not adjust dt
- Examples: Pre-step constraints violated, residual explosion

## Operational Meaning

PhaseLoom provides the "Sensor-Governor" function for the CBTSV1 system:

1. **Sensing**: 27 threads continuously monitor residuals across all domains, scales, and response tiers
2. **Governing**: Gate system enforces LoC at step boundaries; rails provide corrective actions
3. **Orchestrating**: Integration with orchestrator enables regime-based control
4. **Auditing**: Receipt system provides immutable audit trail of all decisions

The system ensures numerical stability while maintaining physical and semantic coherence throughout the simulation.

## Artifacts Generated

- Thread state snapshots
- Rail action lists
- Receipts (attempt, step, window)
- Band metrics (omega, coherence, danger)
- Regime hashes and labels
- Gate validation reports

## Example Pseudocode

```python
from src.phaseloom.phaseloom_27 import PhaseLoom27
from src.phaseloom.phaseloom_memory import PhaseLoomMemory
from src.phaseloom.phaseloom_octaves import PhaseLoomOctaves

# Initialize PhaseLoom system
loom = PhaseLoom27()
memory = PhaseLoomMemory(fields, unified_clock=clock)
octaves = PhaseLoomOctaves(N_threads=27, max_octaves=8)

# Update residuals from solver
loom.update_residual('PHY', 'L', residual_phy)
loom.update_residual('CONS', 'M', residual_cons)
loom.update_residual('SEM', 'H', residual_sem)

# Check gate before step
passed, reasons = loom.check_gate_step()
if not passed:
    return REJECT, reasons

# Get dt cap and dominant thread
dt_cap, dominant = loom.arbitrate_dt()

# Get rails for dominant thread
rails = loom.get_rails(dominant)

# Process octave analysis
omega_sample = compute_omega(residuals)
octave_result = octaves.process_sample(omega_sample)

# Update memory
memory.post_loom_update(octave_result, step)

# Return to stepper
return ACCEPT, dt_cap, rails, dominant
```

This contract ensures PhaseLoom operates as the authoritative governance layer for CBTSV1, maintaining coherence across all simulation scales.
