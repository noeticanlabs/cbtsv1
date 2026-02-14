# CBTSV1 Cleanup Plan - Version 2 (Safe Execution)

**Mission:** Transform cbtsv1 into a purpose-built GR solver + framework repo with clean module boundaries.

---

## Execution Order

### Phase 0: Safety First

| Step | Action | Notes |
|------|--------|-------|
| 0.1 | Create branch/tag: `pre-cleanup-cbtsv1-v0` | Git backup before any changes |
| 0.2 | Update `.gitignore` | Add: `artifacts/`, `*_results.json`, `*_report*.json`, `*.jsonl` |

### Phase 1: Stop Tracking Outputs (Move to Artifacts)

**Rule:** Never delete potential fixtures. Move to artifacts, audit, then decide.

| File | Action | New Location |
|------|--------|--------------|
| `aeonic_receipts.jsonl` | MOVE → artifacts/ | artifacts/aeonic_receipts.jsonl |
| `compiled_nir.json` | MOVE → artifacts/ | artifacts/compiled_nir.json |
| `mms_nir.json` | MOVE → artifacts/ | artifacts/mms_nir.json |
| `compliance_report*.json` | MOVE → artifacts/ | artifacts/ |
| `final_compliance_report.json` | MOVE → artifacts/ | artifacts/ |
| `nsc_*_results.json` | MOVE → artifacts/ | artifacts/ |
| `test_gcat0_5_results.json` | MOVE → artifacts/ | artifacts/ |
| `terminology_registry.json` | MOVE → artifacts/ | artifacts/ |
| `DEBUG_ANALYSIS.md` | MOVE → artifacts/ | artifacts/ |
| `IMPLEMENTATION_SUMMARY_LEMMA3.md` | MOVE → artifacts/ | artifacts/ |

**After move:** Run grep to verify no active code imports these before final deletion.

### Phase 2: Archive Duplicate Doc Spines

| Source | Destination |
|--------|-------------|
| `coherence_spine/` | `archive/docs_legacy/coherence_spine/` |
| `coherence_math_spine/` | `archive/docs_legacy/coherence_math_spine/` |
| `Coherence_Spec_v1_0/` | `archive/docs_legacy/Coherence_Spec_v1_0/` |
| `Technical_Data/` | `archive/docs_legacy/Technical_Data/` |

### Phase 3: Build New Structure (Copy, Don't Move Yet)

Create this tree:
```
src/cbtsv1/
├── __init__.py
├── solvers/
│   └── gr/
│       ├── __init__.py
│       ├── gr_solver.py
│       ├── equations/
│       │   └── rhs.py
│       ├── constraints/
│       │   └── constraints.py
│       ├── gauge/
│       │   └── gauge.py
│       ├── geometry/
│       │   ├── geometry.py
│       │   ├── core_fields.py
│       │   └── geometry_nsc.py
│       ├── stepper.py
│       ├── scheduler.py
│       ├── ledger.py
│       ├── receipts.py
│       ├── gates.py
│       ├── clock.py
│       ├── clocks.py
│       ├── loc.py
│       ├── sem.py
│       ├── coherence.py
│       ├── ttl_calculator.py
│       └── hard_invariants.py
├── numerics/
│   ├── __init__.py
│   ├── spectral/
│   │   ├── __init__.py
│   │   ├── cache.py
│   │   └── spectral_cache.py
│   ├── elliptic/
│   │   ├── __init__.py
│   │   └── solver.py
│   ├── multigrid.py      # from core/nsc_multigrid_solvers.py
│   ├── symplectic.py     # from core/nsc_symplectic_integrators.py
│   └── fd_stabilized.py  # from core/nsc_compact_fd_stabilized.py
└── framework/
    ├── __init__.py
    ├── receipts.py        # from aeonic/aeonic_receipts.py
    ├── memory_bank.py    # from aeonic/aeonic_memory_bank.py
    ├── clocks.py         # from aeonic/aeonic_clocks.py
    ├── memory_contract.py # from aeonic/aeonic_memory_contract.py
    ├── orchestrator.py   # from phaseloom/phaseloom_gr_orchestrator.py
    ├── ledger_receipts.py # from phaseloom/phaseloom_receipts_gr.py
    └── phaseloom_27.py   # from phaseloom/phaseloom_27.py
```

### Phase 3a: Copy Solver Core Files

| Source File | Destination |
|-------------|-------------|
| `src/core/gr_solver.py` | `src/cbtsv1/solvers/gr/gr_solver.py` |
| `src/core/gr_rhs.py` | `src/cbtsv1/solvers/gr/equations/rhs.py` |
| `src/core/gr_constraints.py` | `src/cbtsv1/solvers/gr/constraints/constraints.py` |
| `src/core/gr_gauge.py` | `src/cbtsv1/solvers/gr/gauge/gauge.py` |
| `src/core/gr_geometry.py` | `src/cbtsv1/solvers/gr/geometry/geometry.py` |
| `src/core/gr_core_fields.py` | `src/cbtsv1/solvers/gr/geometry/core_fields.py` |
| `src/core/gr_geometry_nsc.py` | `src/cbtsv1/solvers/gr/geometry/geometry_nsc.py` |
| `src/core/gr_stepper.py` | `src/cbtsv1/solvers/gr/stepper.py` |
| `src/core/gr_scheduler.py` | `src/cbtsv1/solvers/gr/scheduler.py` |
| `src/core/gr_ledger.py` | `src/cbtsv1/solvers/gr/ledger.py` |
| `src/core/gr_receipts.py` | `src/cbtsv1/solvers/gr/receipts.py` |
| `src/core/gr_gates.py` | `src/cbtsv1/solvers/gr/gates.py` |
| `src/core/gr_clock.py` | `src/cbtsv1/solvers/gr/clock.py` |
| `src/core/gr_clocks.py` | `src/cbtsv1/solvers/gr/clocks.py` |
| `src/core/gr_loc.py` | `src/cbtsv1/solvers/gr/loc.py` |
| `src/core/gr_sem.py` | `src/cbtsv1/solvers/gr/sem.py` |
| `src/core/gr_coherence.py` | `src/cbtsv1/solvers/gr/coherence.py` |
| `src/core/gr_ttl_calculator.py` | `src/cbtsv1/solvers/gr/ttl_calculator.py` |
| `src/core/hard_invariants.py` | `src/cbtsv1/solvers/gr/hard_invariants.py` |

**Note on hpc_kernels.py:** Audit import usage first. Move to `archive/optional_hpc/` if unused.

### Phase 3b: Copy Numerics

| Source | Destination |
|--------|-------------|
| `src/spectral/` | `src/cbtsv1/numerics/spectral/` |
| `src/elliptic/` | `src/cbtsv1/numerics/elliptic/` |
| `src/core/nsc_multigrid_solvers.py` | `src/cbtsv1/numerics/multigrid.py` |
| `src/core/nsc_symplectic_integrators.py` | `src/cbtsv1/numerics/symplectic.py` |
| `src/core/nsc_compact_fd_stabilized.py` | `src/cbtsv1/numerics/fd_stabilized.py` |

### Phase 3c: Audit Framework Dependencies

**Before moving framework files, run import audit:**

```bash
# Find all imports from contracts/
grep -r "from src.contracts" --include="*.py" src/

# Find all imports from receipts/
grep -r "from src.receipts" --include="*.py" src/

# Find all imports from aeonic/
grep -r "from src.aeonic" --include="*.py" src/

# Find all imports from phaseloom/
grep -r "from src.phaseloom" --include="*.py" src/
```

**Only copy files that are actually imported by solver code.**

### Phase 3d: Copy Framework Files

| Source File | Destination |
|-------------|-------------|
| `src/aeonic/aeonic_receipts.py` | `src/cbtsv1/framework/receipts.py` |
| `src/aeonic/aeonic_memory_bank.py` | `src/cbtsv1/framework/memory_bank.py` |
| `src/aeonic/aeonic_clocks.py` | `src/cbtsv1/framework/clocks.py` |
| `src/aeonic/aeonic_memory_contract.py` | `src/cbtsv1/framework/memory_contract.py` |
| `src/phaseloom/phaseloom_gr_orchestrator.py` | `src/cbtsv1/framework/orchestrator.py` |
| `src/phaseloom/phaseloom_receipts_gr.py` | `src/cbtsv1/framework/ledger_receipts.py` |
| `src/phaseloom/phaseloom_27.py` | `src/cbtsv1/framework/phaseloom_27.py` |

**Keep src/contracts/ and src/receipts/ in place until import rewiring is complete.**

### Phase 4: Import Rewrite

Update imports in copied files to reference new module paths:

```python
# Old imports in solver code
from .gr_geometry import GRGeometry
from src.phaseloom.phaseloom_gr_orchestrator import GRPhaseLoomOrchestrator

# New imports
from cbtsv1.solvers.gr.geometry import GRGeometry
from cbtsv1.framework.orchestrator import GRPhaseLoomOrchestrator
```

**After import rewrite, run tests to verify solver still works.**

### Phase 5: Archive External Ecosystems

| Source | Destination |
|--------|-------------|
| `lean/` | `archive/lean_theory/` |
| `specifications/` | `archive/specifications/` |
| `proofs_index/` | `archive/proofs_index/` |
| `compliance_tests/` | `archive/compliance_tests/` |

### Phase 5b: Archive Old Plans

| Source | Destination |
|--------|-------------|
| `plans/nsc_*` | `archive/plans_legacy/` |
| `plans/hadamard_*` | `archive/plans_legacy/` |
| `plans/triaxis_*` | `archive/plans_legacy/` |
| `plans/modularization_*` | `archive/plans_legacy/` |

**Keep in root:** `plans/*gr_solver*`, `plans/*minkowski*`, `plans/*hpc_integration*`

### Phase 6: Delete Old Paths

**Only after Phase 4 tests pass:**

| Path | Action |
|------|--------|
| `src/core/gr_*.py` | DELETE ( originals) |
| `src/core/nsc_*.py` | DELETE (if moved to numerics) |
| `src/spectral/` | DELETE (if moved to numerics) |
| `src/elliptic/` | DELETE (if moved to numerics) |
| `src/aeonic/` | DELETE (if moved to framework) |
| `src/phaseloom/` | DELETE (if moved to framework) |

### Phase 7: Create New Docs Spine

Create `docs/solver/gr/`:
- `00_scope.md` - Mission and formulation
- `01_equations.md` - ADM/BSSN equations
- `02_state_and_variables.md` - State vector
- `03_constraints.md` - Hamiltonian/Momentum
- `04_gauge.md` - Gauge conditions
- `05_numerics.md` - Integrator, dissipation
- `06_validation_tests.md` - Test cases

Create `docs/framework/`:
- `receipts.md` - Receipt format
- `ledger_format.md` - Ledger structure
- `conformance.md` - Validation steps
- `cli.md` - Running the solver

### Phase 8: Final Verification

- Run full test suite
- Verify target structure matches specification
- Clean up empty directories

---

## Scripts Triage

| Script | Status | Reason |
|--------|--------|--------|
| `scripts/run_nllc_gr_test.py` | KEEP | Active test runner |
| `scripts/mms_fast_sweep.py` | KEEP | Validation |
| `scripts/mms_parameter_sweep.py` | KEEP | Validation |
| `scripts/inspect_receipts.py` | KEEP | Debugging |
| `scripts/update_imports.py` | KEEP | Import rewiring |
| `scripts/timing_script.py` | KEEP | Profiling |
| `scripts/diagnostic_*.py` | KEEP | Debugging |
| `scripts/run_wp1_nllc_test.py` | ARCHIVE | Not needed |
| `scripts/run_tgs.py` | ARCHIVE | Not needed |
| `scripts/compile_nllc.py` | ARCHIVE | Not needed |

## Tests Triage

| Test Pattern | Status | Reason |
|-------------|--------|--------|
| `test_*mms*.py` | KEEP | Regression tests |
| `test_*gauge*.py` | KEEP | Core functionality |
| `test_*constraint*.py` | KEEP | Core functionality |
| `test_*ricci*.py` | KEEP | Geometry |
| `test_*bianchi*.py` | KEEP | Integrity |
| `test_*gate*.py` | KEEP | Gate policy |
| `test_*nsc*.py` | ARCHIVE | Language tests |
| `test_*nllc*.py` | ARCHIVE | Language tests |
| `test_*parser*.py` | ARCHIVE | Language tests |
| `test_hadamard*.py` | ARCHIVE | Non-solver |
| `generate_health_report.py` | DELETE | Generated output |

## Config Triage

| Config | Status |
|--------|--------|
| `config/gr_*.nsc` | KEEP |
| `config/gr_*.py` | KEEP |
| `config/gate_policy_default.json` | KEEP |
| `config/coupling_policy_v0.1.json` | KEEP |
| `config/golden_run.nscb` | KEEP |
| `config/policy_bundle.nscb` | ARCHIVE |

---

## Key Rules

1. **Never delete potential fixtures** - move to artifacts first
2. **Copy before delete** - build new structure, verify, then remove old
3. **Audit imports first** - verify dependencies before archiving
4. **Test after import rewiring** - ensure solver still works
5. **Keep backup branch** - always have revert path
