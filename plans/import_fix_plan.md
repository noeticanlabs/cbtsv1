# Import Fix Plan

## Summary

The import checker found **76 total issues** in the codebase:

| Category | Count | Fix Required |
|----------|-------|--------------|
| Wrong `cbtsv1.` prefix | 38 | Replace `cbtsv1.` → `src.cbtsv1.` |
| Missing `src.` prefix | 19 | Add `src.` prefix to nsc.* imports |
| Naked imports | 19 | Add `src.` prefix (same as missing src prefix) |

**Note:** The "naked imports" and "missing src prefix" categories overlap - both require adding the `src.` prefix.

---

## Issue Breakdown by File

### Phase 1: Fix Wrong `cbtsv1.` Prefix (38 issues)

These files use `cbtsv1.` but should use `src.cbtsv1.` because `cbtsv1` is a package inside `src/`.

#### 1.1 `src/cbtsv1/solvers/gr/gr_solver.py` (10 issues)
- Line 23: `cbtsv1.framework.phaseloom_gr_orchestrator` → `src.cbtsv1.framework.phaseloom_gr_orchestrator`
- Line 24: `cbtsv1.framework.aeonic_memory_bank` → `src.cbtsv1.framework.aeonic_memory_bank`
- Line 25: `cbtsv1.framework.aeonic_memory_contract` → `src.cbtsv1.framework.aeonic_memory_contract`
- Line 26: `cbtsv1.framework.aeonic_clocks` → `src.cbtsv1.framework.aeonic_clocks`
- Line 27: `cbtsv1.framework.aeonic_receipts` → `src.cbtsv1.framework.aeonic_receipts`
- Line 28: `cbtsv1.framework.phaseloom_27` → `src.cbtsv1.framework.phaseloom_27`
- Line 30: `cbtsv1.numerics.spectral.cache` → `src.cbtsv1.numerics.spectral.cache`
- Line 31: `cbtsv1.solvers.gr.geometry` → `src.cbtsv1.solvers.gr.geometry`
- Line 32: `cbtsv1.solvers.gr.geometry.geometry_nsc` → `src.cbtsv1.solvers.gr.geometry.geometry_nsc`
- Line 33: `cbtsv1.numerics.elliptic.solver` → `src.cbtsv1.numerics.elliptic.solver`

#### 1.2 `src/cbtsv1/solvers/gr/gr_stepper.py` (5 issues)
- Line 15: `cbtsv1.framework.phaseloom_gr_orchestrator` → `src.cbtsv1.framework.phaseloom_gr_orchestrator`
- Line 16: `cbtsv1.framework.aeonic_memory_bank` → `src.cbtsv1.framework.aeonic_memory_bank`
- Line 17: `cbtsv1.framework.aeonic_clocks` → `src.cbtsv1.framework.aeonic_clocks`
- Line 18: `cbtsv1.framework.aeonic_receipts` → `src.cbtsv1.framework.aeonic_receipts`
- Line 19: `cbtsv1.solvers.gr.gates` → `src.cbtsv1.solvers.gr.gates`

#### 1.3 `src/cbtsv1/framework/phaseloom_gr_orchestrator.py` (12 issues)
- Line 24: `cbtsv1.framework.aeonic_clocks` → `src.cbtsv1.framework.aeonic_clocks`
- Line 25: `cbtsv1.framework.aeonic_memory_bank` → `src.cbtsv1.framework.aeonic_memory_bank`
- Line 26: `cbtsv1.framework.aeonic_receipts` → `src.cbtsv1.framework.aeonic_receipts`
- Line 27: `cbtsv1.framework.aeonic_memory_contract` → `src.cbtsv1.framework.aeonic_memory_contract`
- Line 28: `cbtsv1.solvers.gr.logging_config` → `src.cbtsv1.solvers.gr.logging_config`
- Line 30: `cbtsv1.solvers.gr.phases` → `src.cbtsv1.solvers.gr.phases`
- Line 40: `cbtsv1.solvers.gr.scheduler` → `src.cbtsv1.solvers.gr.scheduler`
- Line 41: `cbtsv1.solvers.gr.sem` → `src.cbtsv1.solvers.gr.sem`
- Line 43: `cbtsv1.solvers.gr.ttl_calculator` → `src.cbtsv1.solvers.gr.ttl_calculator`
- Line 44: `cbtsv1.framework.receipt_schemas` → `src.cbtsv1.framework.receipt_schemas`
- Line 45: `cbtsv1.framework.orchestrator_contract_memory` → `src.cbtsv1.framework.orchestrator_contract_memory`
- Line 205: `cbtsv1.numerics.spectral.cache` → `src.cbtsv1.numerics.spectral.cache`

#### 1.4 Other files with wrong `cbtsv1.` prefix (11 issues)
- `src/cbtsv1/framework/phaseloom_threads_gr.py` (line 175): `cbtsv1.solvers.gr.geometry.core_fields` → `src.cbtsv1.solvers.gr.geometry.core_fields`
- `src/cbtsv1/framework/aeonic_memory_bank.py` (line 4): `cbtsv1.framework.aeonic_clocks` → `src.cbtsv1.framework.aeonic_clocks`
- `src/cbtsv1/framework/aeonic_memory_bank.py` (line 5): `cbtsv1.framework.aeonic_receipts` → `src.cbtsv1.framework.aeonic_receipts`
- `src/cbtsv1/framework/aeonic_memory_contract.py` (line 2): `cbtsv1.framework.aeonic_memory_bank` → `src.cbtsv1.framework.aeonic_memory_bank`
- `src/cbtsv1/framework/aeonic_memory_contract.py` (line 3): `cbtsv1.framework.receipt_schemas` → `src.cbtsv1.framework.receipt_schemas`
- `src/cbtsv1/framework/aeonic_memory_contract.py` (line 4): `cbtsv1.solvers.gr.ttl_calculator` → `src.cbtsv1.solvers.gr.ttl_calculator`
- `src/cbtsv1/framework/aeonic_memory_contract.py` (line 5): `cbtsv1.solvers.gr.gates` → `src.cbtsv1.solvers.gr.gates`
- `src/cbtsv1/framework/aeonic_receipts.py` (line 5): `cbtsv1.framework.receipt_schemas` → `src.cbtsv1.framework.receipt_schemas`
- `src/cbtsv1/framework/aeonic_receipts.py` (line 28): `cbtsv1.framework.receipt_schemas` → `src.cbtsv1.framework.receipt_schemas`
- `src/cbtsv1/numerics/fd_stabilized.py` (line 477): `cbtsv1.solvers.gr.geometry` → `src.cbtsv1.solvers.gr.geometry`
- `src/cbtsv1/numerics/fd_stabilized.py` (line 530): `cbtsv1.solvers.gr.geometry.geometry_nsc` → `src.cbtsv1.solvers.gr.geometry.geometry_nsc`
- `src/cbtsv1/solvers/gr/gr_sem.py` (line 16): `cbtsv1.framework.receipt_schemas` → `src.cbtsv1.framework.receipt_schemas`
- `src/cbtsv1/solvers/gr/defined_coherence_blocks.py` (line 27): `cbtsv1.vendor.coherence_framework.coherence.core` → `src.cbtsv1.vendor.coherence_framework.coherence.core`
- `src/cbtsv1/solvers/gr/gr_ledger.py` (line 6): `cbtsv1.solvers.gr.receipts` → `src.cbtsv1.solvers.gr.receipts`
- `src/cbtsv1/solvers/gr/constraints/constraints.py` (line 57): `cbtsv1.solvers.gr.geometry.core_fields` → `src.cbtsv1.solvers.gr.geometry.core_fields`
- `src/cbtsv1/solvers/gr/constraints/constraints.py` (line 58): `cbtsv1.numerics.elliptic.solver` → `src.cbtsv1.numerics.elliptic.solver`

---

### Phase 2: Fix Missing `src.` Prefix for nsc.* Imports (19 issues)

These files import from `nsc.*` but should use `src.nsc.*` because `nsc` is a package inside `src/`.

#### 2.1 `src/nsc/compat.py` (15 issues)
- Line 10: `nsc.compat` → `src.nsc.compat`
- Line 11: `nsc.domains.geometry.gr` → `src.nsc.domains.geometry.gr`
- Line 20: `nsc.domains.algebra.tensor` → `src.nsc.domains.algebra.tensor`
- Line 24: `nsc.models.ledger` → `src.nsc.models.ledger`
- Line 28: `nsc.domains.numerical.stencils` → `src.nsc.domains.numerical.stencils`
- Line 32: `nsc.models.exec.vm` → `src.nsc.models.exec.vm`
- Line 36: `nsc.domains.numerical.quadrature` → `src.nsc.domains.numerical.quadrature`
- Line 68: `nsc.domains.geometry.gr` → `src.nsc.domains.geometry.gr`
- Line 71: `nsc.domains.fluids.navier_stokes` → `src.nsc.domains.fluids.navier_stokes`
- Line 74: `nsc.domains.geometry.ym` → `src.nsc.domains.geometry.ym`
- Line 77: `nsc.models.ledger` → `src.nsc.models.ledger`
- Line 78: `nsc.models.exec.vm` → `src.nsc.models.exec.vm`
- Line 96: `nsc.domains.algebra.tensor` → `src.nsc.domains.algebra.tensor`
- Line 101: `nsc.models.ledger` → `src.nsc.models.ledger`
- Line 106: `nsc.domains.numerical.stencils` → `src.nsc.domains.numerical.stencils`
- Line 111: `nsc.models.exec.vm` → `src.nsc.models.exec.vm`
- Line 116: `nsc.domains.numerical.quadrature` → `src.nsc.domains.numerical.quadrature`

#### 2.2 `src/nsc/domains/geometry/geometry.py` (1 issue)
- Line 21: `nsc.types` → `src.nsc.types`

#### 2.3 `src/nsc/domains/fluids/navier_stokes.py` (1 issue)
- Line 22: `nsc.types` → `src.nsc.types`

---

## Fix Patterns

### Pattern 1: Replace wrong `cbtsv1.` prefix
```python
# Before
from cbtsv1.framework.phaseloom_gr_orchestrator import GRPhaseLoomOrchestrator

# After
from src.cbtsv1.framework.phaseloom_gr_orchestrator import GRPhaseLoomOrchestrator
```

### Pattern 2: Add missing `src.` prefix for nsc imports
```python
# Before
from nsc.domains.geometry.gr import GRDomain

# After
from src.nsc.domains.geometry.gr import GRDomain
```

---

## Recommended Execution Order

### Phase 1: Fix wrong `cbtsv1.` prefix
**Rationale:** These are in the `src/cbtsv1/` package which is a sub-package. Changing them first establishes the correct import pattern.

1. Fix `src/cbtsv1/solvers/gr/gr_solver.py` (10 issues)
2. Fix `src/cbtsv1/solvers/gr/gr_stepper.py` (5 issues)
3. Fix `src/cbtsv1/framework/phaseloom_gr_orchestrator.py` (12 issues)
4. Fix remaining files in `src/cbtsv1/` (11 issues)

**Validation:** Run `python scripts/check_imports.py` - should reduce issues from 76 to 38.

### Phase 2: Fix missing `src.` prefix for nsc imports
**Rationale:** The nsc package is used by compat.py which may be imported early.

1. Fix `src/nsc/compat.py` (15 issues)
2. Fix `src/nsc/domains/geometry/geometry.py` (1 issue)
3. Fix `src/nsc/domains/fluids/navier_stokes.py` (1 issue)

**Validation:** Run `python scripts/check_imports.py` - should show 0 issues.

---

## Validation Steps

After each phase, run:

```bash
python scripts/check_imports.py
```

Expected results:
- After Phase 1: 38 remaining issues (all missing src. prefix for nsc)
- After Phase 2: 0 issues

---

## Alternative: Use sed for Bulk Fixes

For faster execution, you can use sed to apply fixes:

### Fix wrong cbtsv1. prefix:
```bash
# In src/cbtsv1 directory
find . -name "*.py" -exec sed -i 's/from cbtsv1\./from src.cbtsv1./g' {} \;
find . -name "*.py" -exec sed -i 's/import cbtsv1\./import src.cbtsv1./g' {} \;
```

### Fix missing src. prefix for nsc:
```bash
# In src/nsc directory  
find . -name "*.py" -exec sed -i 's/from nsc\./from src.nsc./g' {} \;
find . -name "*.py" -exec sed -i 's/import nsc\./import src.nsc./g' {} \;
```

---

## Notes

1. The import checker considers these valid packages: `aml`, `hadamard`, `host_api`, `nllc`, `nsc`, `cbtsv1`
2. All imports within `src/` should use the `src.` prefix to work correctly
3. The "naked imports" and "missing src prefix" categories overlap - they represent the same fix needed
