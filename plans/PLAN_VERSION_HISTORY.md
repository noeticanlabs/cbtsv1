# Plan Document Version History

## Overview
This document tracks the version history of plan documents as the Noetica/NSC/Triaxis system evolves.

---

## NSC Upgrade Plan

| Version | File | Status | Date |
|---------|------|--------|------|
| v1.0 | `plans/nsc_upgrade_plan.md` | ORIGINAL | - |
| **v2.0** | `plans/nsc_upgrade_plan_v2.md` | **CURRENT** | 2026-01-27 |

**v2.0 Changes:**
- Added implementation status tables
- Documented TypeChecker completion
- Documented Mem2Reg pass
- Marked ConstantFolding/DeadCodeElimination as pending
- Listed new files added

---

## NSC Split Plan

| Version | File | Status | Date |
|---------|------|--------|------|
| v1.0 | `plans/nsc_split_plan.md` | ORIGINAL | - |
| **v2.0** | `plans/nsc_split_plan_v2.md` | **CURRENT** | 2026-01-27 |

**v2.0 Changes:**
- Added implementation status tables
- Documented 4-byte bytecode format
- Documented Hadamard opcode table
- Listed NSC→Hadamard compiler
- Marked JIT hooks and constant folding as pending

---

## PhaseLoom GR/NR Integration Plan

| Version | File | Status | Date |
|---------|------|--------|------|
| v1.0 | `plans/nsc_phaseloom_gr_nr_integration_plan.md` | ORIGINAL | - |
| **v2.0** | `plans/nsc_phaseloom_gr_nr_integration_plan_v2.md` | **CURRENT** | 2026-01-27 |

**v2.0 Changes:**
- Added Host API implementation status
- Documented PhaseLoom 27-thread lattice
- Listed receipt schema
- Marked NSC_GR dialect as pending
- Documented troubleshooting resolutions

---

## Summary of Implementation Status

| Plan | Original Tasks | Completed | Pending |
|------|---------------|-----------|---------|
| NSC Upgrade | 5 | 4 (80%) | 1 (CLI tooling) |
| NSC Split | 5 | 4 (80%) | 1 (JIT/benchmarks) |
| PhaseLoom Integration | 4 | 3 (75%) | 1 (NSC_GR dialect) |

**Overall: ~80% implemented**

---

## Next Planned Versions

### v3.0 (Future)
- Document completion of remaining pending items
- Add benchmark results
- Document performance metrics
- Finalize deprecation of legacy code

---

## Related Documents

- [`terminology_registry.json`](terminology_registry.json) — Machine-readable ontology
- [`compliance_tests/ci_check.py`](compliance_tests/ci_check.py) — CI enforcement
- [`tests/test_full_stack_integration.py`](tests/test_full_stack_integration.py) — 46 integration tests
