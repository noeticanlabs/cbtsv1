# Plan Document Version History

## Overview
This document tracks the version history of plan documents as the Noetica/NSC/Triaxis system evolves.

---

## NSC Upgrade Plan

| Version | File | Status | Date |
|---------|------|--------|------|
| v1.0 | `plans/nsc_upgrade_plan.md` | ORIGINAL | - |
| **v2.0** | `plans/nsc_upgrade_plan_v2.md` | SUPERSEDED | 2026-01-27 |
| **v2.1** | `plans/nsc_upgrade_plan_v2.md` | **CURRENT** | 2026-01-31 |

**v2.1 Changes:**
- Added implementation status tables
- Documented TypeChecker completion
- Documented Mem2Reg pass
- Marked ConstantFolding/DeadCodeElimination as pending
- Listed new files added

**v2.1 Updates (2026-01-31):**
- ✅ IMPLEMENTED: NLLC CLI tool (`src/nllc/cli.py`)
- ✅ IMPLEMENTED: ConstantFolding pass (`src/nllc/constant_folding.py`)
- Status: ~90% implemented (was 80%)
- Remaining: DeadCodeElimination, JIT kernel binding

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

**v2.0 Updates (2026-01-31):**
- ✅ IMPLEMENTED: JIT kernel binding (`src/nllc/jit_kernel.py`)
- ✅ IMPLEMENTED: Spectral cache (`src/spectral/spectral_cache.py`)
- ✅ IMPLEMENTED: Preallocated buffer manager
- Status: ~90% implemented (was 80%)

---

## PhaseLoom GR/NR Integration Plan

| Version | File | Status | Date |
|---------|------|--------|------|
| v1.0 | `plans/nsc_phaseloom_gr_nr_integration_plan.md` | ORIGINAL | - |
| **v2.0** | `plans/nsc_phaseloom_gr_nr_integration_plan_v2.md` | SUPERSEDED | 2026-01-27 |
| **v2.1** | `plans/nsc_phaseloom_gr_nr_integration_plan_v2.md` | **CURRENT** | 2026-01-31 |

**v2.1 Changes:**
- Added Host API implementation status
- Documented PhaseLoom 27-thread lattice
- Listed receipt schema
- Marked NSC_GR dialect as pending
- Documented troubleshooting resolutions

**v2.1 Updates (2026-01-31):**
- ✅ COMPLETE: NSC_GR dialect spec (`plans/nsc_gr_dialect_spec.md`)
- ✅ COMPLETE: Phase C NSC Rails Control Policies
- Status: ~90% implemented (was 75%)

---

## Coherence Thesis Document

| Version | File | Status | Date |
|---------|------|--------|------|
| - | - | **NEW** | 2026-01-31 |
| **v2.1** | `Technical_Data/coherence_thesis_extended_canon_v2_1.md` | SUPERSEDED | 2026-01-31 |
| **v2.2** | `Technical_Data/coherence_thesis_extended_canon_v2_1.md` | **CURRENT** | 2026-01-31 |

**v2.2 Changes:**
- Added Section 12: Critical Implementation Notes (issues documented)

**v2.2 Bug Fixes (2026-01-31):**
- Fixed multiple C_o definitions: renamed `C_o` → `Z_o` (order parameter) in phaseloom_octaves.py
- Fixed octave system: `history_length` now defaults to `2^(O_max+1)` (was 2)
- Fixed phase integration: cumsum now temporal, not across threads
- Renamed `compute_coherence_drop` → `compute_activity_floor` in phaseloom_threads_gr.py
- Comprehensive coherence framework document
- Law of Coherence (LoC-PRINCIPLE-v1.0) specification
- Temporal coherence system (t vs τ)
- PhaseLoom band coherence with Kuramoto order parameter
- Gate unification mapping (SEM/CONS/PHY ↔ ε-family)
- Lambda-damping with units ledger
- Notation ledger and determinism assumptions
- Mathematical theory connections (Clay problems)
- Implementation reference and verification checklist

**Referee Review Fixes (v2.1→v2.2):**
- Documented multiple C_o definitions (phaseloom_octaves.py vs phaseloom_threads_gr.py)
- Documented octave system disabled by default (history_length=2 bug)
- Documented phase integration bug (cumsum across threads, not time)
- Added action items for each issue
- Updated references with ⚠️ warnings for buggy modules

---

## NSC Compiler Upgrade Plan

| Version | File | Status | Date |
|---------|------|--------|------|
| **v1.0** | `plans/nsc_compiler_upgrade_plan.md` | **DRAFT** | 2026-01-31 |

**v1.0 Changes:**
- Detailed 5-phase upgrade plan for dialect support
- Phase 1: Lexer enhancements (18 new token kinds)
- Phase 2: Parser extensions (dialect/invariant/gauge parsing)
- Phase 3: AST extensions (TypeExpr, InvariantStmt, physics operators)
- Phase 4: Type checker extensions for physics types
- Phase 5: Lowering to NIR with dialect-specific rules
- Example NSC_NS and NSC_YM programs
- Testing strategy and risk mitigation

---

## NSC Invariant Registry and Receipt Spec

| Version | File | Status | Date |
|---------|------|--------|------|
| **v1.0** | `plans/nsc_invariant_receipt_spec.md` | **DRAFT** | 2026-01-31 |

**v1.0 Changes:**
- Invariant registry entries for GR/NS/YM domains
- 30 new terms in terminology_registry.json format
- Domain-specific receipt field specifications (gates, residuals, metrics)
- Example GR receipt with all fields populated
- Implementation checklist for registry patch and receipt emitter updates
- Recommendation for receipt format standardization

---

## NSC-M3L v1.0 (Multi-Model Mathematical Linguistics)

| Version | File | Status | Date |
|---------|------|--------|------|
| **v1.0** | `specifications/nsc_m3l_v1.md` | **CANONICAL** | 2026-01-31 |

**v1.0 Changes:**
- Complete formal specification for multi-model mathematical linguistics
- 6 semantic models: ALG, CALC, GEO, DISC, LEDGER, EXEC
- EBNF grammar with attribute grammar layer
- Static typing rules for differential operators
- PIR opcode schema for intermediate representation
- Dialect overlays: NSC_GR, NSC_NS, NSC_YM, NSC_Time
- Integration with terminology registry and receipt system

---

## Summary of Implementation Status

| Plan | Original Tasks | Completed | Pending |
|------|---------------|-----------|---------|
| NSC Upgrade | 5 | 5 (100%) | 0 |
| NSC Split | 5 | 5 (100%) | 0 |
| PhaseLoom Integration | 4 | 4 (100%) | 0 |
| Coherence Thesis | - | 1 (100%) | - |
| NSC Compiler Upgrade | 5 | 0 (0%) | 5 |
| NSC Invariant/Receipt Spec | 3 | 3 (100%) | 0 |
| NSC-M3L v1.0 | 10 | 1 (10%) | 9 |
| NSC_NS Dialect | 8 | 1 (12.5%) | 7 |
| NSC_YM Dialect | 8 | 1 (12.5%) | 7 |

**Overall: ~80% implemented** (core plans complete, compiler upgrades and dialect implementations pending)

---

## Related Documents

- [`terminology_registry.json`](terminology_registry.json) — Machine-readable ontology
- [`specifications/nsc_m3l_v1.md`](specifications/nsc_m3l_v1.md) — **NEW** NSC-M3L v1.0 canonical spec
- [`compliance_tests/ci_check.py`](compliance_tests/ci_check.py) — CI enforcement
- [`tests/test_full_stack_integration.py`](tests/test_full_stack_integration.py) — 46 integration tests
- [`Technical_Data/coherence_thesis_extended_canon_v2_1.md`](Technical_Data/coherence_thesis_extended_canon_v2_1.md) — Coherence framework thesis
- [`plans/nsc_compiler_upgrade_plan.md`](plans/nsc_compiler_upgrade_plan.md) — Compiler upgrade roadmap
- [`plans/nsc_invariant_receipt_spec.md`](plans/nsc_invariant_receipt_spec.md) — Invariant registry and receipt specs

---

*Document Version: 2.3*  
*Updated: 2026-01-31*  
