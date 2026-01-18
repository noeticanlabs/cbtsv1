# LoC-Time Integration Templates for PHY, CONS, ORCH

## PHY (Physical) Domain Integration Template

Integrate LoC-Time with Physical domain:
- Update physical time t
- Check CFL and advection constraints

```python
def phy_integration_template(time_state, dt_arbitrated, residuals, drifts):
    # Update t
    time_state.t += dt_arbitrated
    time_state.n += 1
    # Example constraint check
    cfl_margin = 1.0 - (dt_arbitrated / 0.1)  # Assume max dt 0.1
    if cfl_margin < 0:
        return False, 'cfl_violation'
    return True, None
```

## CONS (Constraints) Domain Integration Template

Integrate LoC-Time with Constraints domain:
- Enforce constraint margins
- Update coherence time if margins ok

```python
def cons_integration_template(time_state, dt_arbitrated, margins, eps_H, eps_M):
    m_star = min(margins.values()) if margins else 0.0
    if m_star < 0:
        return False, 'margin_violation'
    # Update tau
    delta_tau = 0.5 * dt_arbitrated  # Simplified
    time_state.tau += delta_tau
    return True, None
```

## ORCH (Orchestrator) Domain Integration Template

Integrate LoC-Time with Orchestrator domain:
- Coordinate multi-clock arbitration
- Log audit trail

```python
def orch_integration_template(time_state, proposals, limiting_clock, audit_info):
    # Example: Log dominant clock
    audit_info['limiting_clock'] = limiting_clock
    audit_info['proposals_count'] = len(proposals)
    # Assume success
    return True, None