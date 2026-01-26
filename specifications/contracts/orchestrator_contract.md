# Orchestrator Contract Specification

## Overview
The orchestrator contract defines the mandatory interface and behavioral rules for the orchestrator component, which aggregates accepted step receipts into canonical windows, decides regime transitions based on coherence and performance metrics, and issues promotions or quarantines. The orchestrator never decides physics but enforces canon by aggregating only accepted history and never rejected attempts. It evaluates windows against thresholds and margins to label operational regimes: stable, constraint-risk, semantic-risk, or perf-risk.

Violations of this contract constitute SEM-hard failures, halting the system and requiring manual intervention.

## Lexicon Declarations
This specification imports the following from the project lexicon (canon v1.2):
- **LoC_axiom**: Law of Coherence (primary invariant)
- **UFE_core**: Universal Field Equation evolution operator
- **CTL_time**: Control & time geometry layer
- **GR_dyn**: General Relativity dynamics

## Entities and Notations
- **M_step**: Accepted step receipt from stepper, containing residuals (ε_H, ε_M), dt used, τ advance, acceptance status
- **W**: Window definition (dictionary specifying either 'num_steps': int or 'delta_tau': float for aggregation window)
- **Performance counters**: Dictionary of metrics (e.g., 'cpu_time_per_step': float, 'memory_peak': float, 'convergence_rate': float)
- **M_orch**: Orchestrator window receipt, aggregating statistics over the window (e.g., mean residuals, regime stats)
- **Regime label**: Categorical label for operational regime ('stable', 'constraint-risk', 'semantic-risk', 'perf-risk')
- **Promotions/Quarantines**: List of actions (e.g., [{'action': 'promote', 'target': 'state_X'}, {'action': 'quarantine', 'reason': 'constraint-risk'}])
- **Thresholds and Margins**: Numerical limits for claims (e.g., residual_threshold = 1e-4, margin = 1e-5; claim "PASS because residual < residual_threshold - margin")

## Inputs
- `accepted_step_receipts`: List of M_step objects (only accepted steps, no rejected attempts)
- `window_definition`: W dictionary defining aggregation window
- `performance_counters`: Dictionary of current performance metrics

## Outputs
- `window_receipt`: M_orch object summarizing window statistics and regime
- `promotions_quarantines`: List of promotion/quarantine actions
- `regime_label`: String label for the current regime

## Behavioral Rules

### Hard Rules
1. Never aggregate rejected attempts into canon stats (unless explicitly labeled as rejected in the receipt).
2. Never declare tests verified without enough accepted history (e.g., minimum accepted steps > history_threshold).
3. Must include thresholds and margins in claims (e.g., "PASS because residual < residual_threshold - margin").

### Additional Rules
- Aggregation must only use M_step from accepted receipts; rejected receipts are ignored or labeled separately.
- Regime evaluation:
  - **Stable**: All window-averaged residuals < thresholds - margins, performance within norms.
  - **Constraint-risk**: Constraints nearing limits (e.g., ε_H > threshold - margin).
  - **Semantic-risk**: Physics invariants violated or semantic checks fail.
  - **Perf-risk**: Performance degradation (e.g., convergence_rate < perf_threshold).
- Issue promotions if regime stable; quarantines if risk detected, marking states for review.
- Claims must quantify with thresholds and margins (no vague "good" or "bad").

## Failure Modes
- **SEM-hard failure**: Aggregating rejected receipts without label, or verifying without history → Halt with detailed error.
- **Regime mislabel**: Incorrect regime without evidence → Soft fail, log and retry.

## Operational Meaning
The orchestrator maintains canon integrity through windowed decision-making, separating orchestration from physics computation. It ensures only coherent, verified evolutions contribute to regime transitions.

## Artifacts Generated
- Window receipts (M_orch) for audit trails
- Promotion/quarantine logs for state management
- Regime labels with quantified justifications

## Example Pseudocode
```python
def orchestrate_window(accepted_receipts, window_def, perf_counters):
    # Hard rule checks
    rejected = [r for r in accepted_receipts if not r.accepted]
    if rejected:
        if not all(r.labeled_rejected for r in rejected):
            return SEM_FAILURE, "Rejected receipts not labeled"
    
    if len(accepted_receipts) < min_accepted_history:
        return SEM_FAILURE, "Insufficient accepted history for verification"
    
    # Aggregate accepted stats
    window_stats = aggregate_accepted_stats(accepted_receipts, window_def)
    
    # Evaluate regime
    regime = evaluate_regime(window_stats, perf_counters, thresholds, margins)
    
    # Generate claims with thresholds/margins
    claims = generate_claims(window_stats, thresholds, margins)  # e.g., "PASS: residual < 1e-4 - 1e-5"
    
    # Decide actions
    actions = []
    if regime == 'stable':
        actions.append({'action': 'promote', 'target': 'canon'})
    else:
        actions.append({'action': 'quarantine', 'reason': regime})
    
    m_orch = WindowReceipt(window_stats, regime, claims)
    return m_orch, actions, regime
```

This contract ensures orchestration remains decoupled from physics, focusing on canon and regime decisions.