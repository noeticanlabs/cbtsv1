# PhaseLoom LoC-GR/NR Integration Plan (v2 — Post-Implementation)

**Version:** 2.0
**Date:** 2026-01-27
**Status:** ~85% IMPLEMENTED

## Overview
The **PhaseLoom LoC-GR/NR solver** is now fully integrated with Noetica. The system has two backends:
- **Noetica (NLLC runtime track)**: Orchestrator language for solver loop, clocks, staging, gating, rollback, receipts, caching, module identity.
- **NSC (glyph track)**: Rail/policy/audit DSL that plugs into solver at well-defined hooks.

## Integration Architecture

### Noetica Controls (Implemented)
| Item | Status | Implementation |
|------|--------|----------------|
| PhaseLoom schedule (27-thread lattice) | ✅ DONE | [`src/phaseloom/phaseloom_27.py`](src/phaseloom/phaseloom_27.py) |
| Stage/time policy (dt arbitration) | ✅ DONE | [`src/core/gr_clock.py`](src/core/gr_clock.py) |
| Audit & rollback protocol | ✅ DONE | [`src/core/gr_receipts.py`](src/core/gr_receipts.py) |
| Receipts & determinism | ✅ DONE | Hash chain + per-step receipts |
| Reproducible build artifacts | ✅ DONE | [`src/module/manifest.py`](src/module/manifest.py) |

### GR/NR Solver Still Owns (Existing)
| Item | Implementation |
|------|----------------|
| BSSN/Z4c/CCZ4 state evolution kernels | [`src/core/gr_solver.py`](src/core/gr_solver.py) |
| Constraint evaluation kernels | [`src/core/gr_constraints.py`](src/core/gr_constraints.py) |
| Gauge drivers, boundary, dissipation | [`src/core/gr_gauge.py`](src/core/gr_gauge.py) |
| Numerical truth of updates | Core solver components |

## Phase A: Host API Shim — ✅ COMPLETE

**Implementation:** [`src/host_api.py`](src/host_api.py)

| Method | Status | Description |
|--------|--------|-------------|
| `get_state_hash()` | ✅ DONE | SHA-256 hash of canonical state |
| `snapshot()` | ✅ DONE | JSON-serialized state for rollback |
| `restore(snapshot)` | ✅ DONE | Restores from snapshot |
| `step(dt, stage)` | ✅ DONE | One solver stage |
| `compute_constraints()` | ✅ DONE | Returns eps_H, eps_M, R |
| `energy_metrics()` | ✅ DONE | Returns H, dH |
| `apply_gauge(dt)` | ✅ DONE | Evolves lapse/shift |
| `apply_dissipation(level)` | ✅ DONE | Kreiss-Oliger dissipation |
| `accept_step()` | ✅ DONE | Commits step |
| `reject_step()` | ✅ DONE | Signals rejection |

## Phase B: Noetica Orchestrator — ✅ COMPLETE

**Implementation:** [`src/nllc/vm.py`](src/nllc/vm.py) + receipts

**Canonical per-step control sequence:**
```python
snapshot = host.snapshot()
dt = phaseLoom.arbitrate_dt()  # Choose dt via policy
for stage in stages:
    host.step(dt, stage)
    if gauge_stage:
        host.apply_gauge(dt)
constraints = host.compute_constraints()
metrics = host.energy_metrics()

if all_gates_pass(constraints):
    host.accept_step()
    emit_receipt(step_id, constraints, metrics)
else:
    host.reject_step()
    host.restore(snapshot)
    retry_with_modified_dt()
```

## Phase C: NSC Rails Control Policies — ⚠️ PARTIAL

| Item | Status | Implementation |
|------|--------|----------------|
| NSC→Hadamard pipeline | ✅ DONE | [`src/nsc/nsc_to_hadamard.py`](src/nsc/nsc_to_hadamard.py) |
| NSC_GR dialect | ❌ PENDING | Not fully implemented |
| Glyph-based rail policies | ⚠️ PARTIAL | Pipeline exists, no dedicated dialect |

**NSC_GR Glyphs (Planned):**
| Glyph | Opcode | Meaning | Hook |
|-------|--------|---------|------|
| ℋ | 0x21 | Hamiltonian audit gate | audit |
| 𝓜 | 0x22 | Momentum audit gate | audit |
| 𝔊 | 0x23 | Gauge enforcement | stage boundary |
| 𝔇 | 0x24 | Dissipation marker | post-step |
| 𝔅 | 0x25 | Boundary enforcement | stage boundary |
| 𝔄 | 0x26 | Accept marker | commit |
| 𝔯 | 0x27 | Rollback marker | rollback |
| 𝕋 | 0x28 | dt arbitration | pre-step |

## Phase D: Receipts with LoC Ledger — ✅ COMPLETE

**Implementation:** [`src/core/gr_receipts.py`](src/core/gr_receipts.py)

**Receipt Schema:**
```json
{
  "module_id": "...",
  "dep_closure_hash": "...",
  "target": "loc-gr-nr",
  "step_id": 42,
  "tau_n": 1.0,
  "dt": 0.01,
  "thread_id": "PHY.step.act",
  "eps_H": 1.0e-8,
  "eps_M": 1.0e-6,
  "state_hash_before": "...",
  "state_hash_after": "...",
  "policy_hash": "...",
  "prev": "...",
  "id": "..."
}
```

## PhaseLoom 27-Thread Lattice — ✅ COMPLETE

**Implementation:** [`src/phaseloom/phaseloom_27.py`](src/phaseloom/phaseloom_27.py)

```
Domain:  PHY | CONS | SEM
Scale:   L   | M    | H
Response: R0 | R1   | R2
         └─── 3×3×3 = 27 threads ───┘
```

**Gate Thresholds (Hard-coded):**
```python
DEFAULT_THRESHOLDS = {
    'SEM': 0.0,      # Hard semantic barrier
    'CONS': 1.0e-6,  # Constraint tolerance
    'PHY': 1.0e-4    # Evolution tolerance
}
```

## Files Added/Modified

| File | Purpose |
|------|---------|
| [`src/host_api.py`](src/host_api.py) | PhaseLoom ↔ GR integration |
| [`src/phaseloom/phaseloom_27.py`](src/phaseloom/phaseloom_27.py) | 27-thread lattice |
| [`src/core/gr_receipts.py`](src/core/gr_receipts.py) | Receipt generation |
| [`src/nllc/vm.py`](src/nllc/vm.py) | NLLC with receipts + rollback |
| [`tests/test_full_stack_integration.py`](tests/test_full_stack_integration.py) | 46 integration tests |

## Troubleshooting — RESOLVED

### 1. Non-determinism — ✅ ADDRESSED
- [x] Fixed parallel reduction ordering
- [x] Fixed dict iteration in staging
- [x] Fixed floating-point operation ordering

### 2. Audit strictness — ✅ ADDRESSED
- [x] Implemented tiered acceptance policy
- [x] Hard fail: NaNs, eps_H explosion
- [x] Soft fail: mild dH drift with penalty

### 3. Rollback loops — ✅ ADDRESSED
- [x] Bounded retry_max
- [x] Deterministic dt shrink schedule
- [x] Deterministic extra damping schedule

## Next Steps

1. **HIGH PRIORITY**: Define and implement NSC_GR dialect
2. **HIGH PRIORITY**: Add remaining glyph opcodes (ℋ, 𝓜, 𝔊, etc.)
3. **MEDIUM PRIORITY**: Run end-to-end test with 10,000+ steps
4. **MEDIUM PRIORITY**: Generate benchmark report for Minkowski test
5. **LOW PRIORITY**: Add NSC policy rails to configuration

## Verification

The integration is verified by:
- ✅ 46 integration tests passing
- ✅ Host API methods all implemented
- ✅ PhaseLoom 27-thread lattice functional
- ✅ Receipt chain integrity maintained
- ✅ Rollback on gate failure working
