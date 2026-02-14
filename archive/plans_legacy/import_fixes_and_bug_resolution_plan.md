# Import Issues and Bug Resolution Plan

## Executive Summary

This plan addresses the import issues and bugs caused by the partial system reorganization. The current state shows:

- **Duplicate files**: `gr_solver/` contains only 3 files while `src/core/` has 25+ actual source files
- **Missing references**: Several files reference `gr_solver/gr_constraints.py`, `gr_solver/gr_geometry.py`, etc. which don't exist
- **86 import statements** across the codebase need to be verified and fixed
- **Incomplete migration**: The reorganization from `gr_solver/` to `src/core/` was partially completed

## Current State Analysis

### Directory Structure Comparison

| Directory | File Count | Purpose |
|-----------|------------|---------|
| `gr_solver/` | 3 files | Backward compatibility stubs |
| `src/core/` | 25+ files | Actual source code |
| `gr_solver/__init__.py` | Re-exports from `src.core` | Backward compatibility |

### Files in `gr_solver/` (Stub Directory)
- `__init__.py` - Re-exports from `src.core`
- `gr_core_fields.py` - DUPLICATE of `src/core/gr_core_fields.py`
- `host_api.py` - Original file, may need migration

### Files Referenced But Missing from `gr_solver/`
- `gr_solver/gr_constraints.py` - References exist, file missing
- `gr_solver/gr_geometry.py` - References exist, file missing
- `gr_solver/gr_stepper.py` - References exist, file missing
- `gr_solver/gr_clock.py` - References exist, file missing
- `gr_solver/spectral/cache.py` - References exist, file missing
- `gr_solver/phaseloom_octaves.py` - References exist, file missing
- `gr_solver/elliptic/solver.py` - References exist, file missing

---

## Problem Categories

### 1. Duplicate Files
- `gr_solver/gr_core_fields.py` duplicates `src/core/gr_core_fields.py`

### 2. Missing Symlinks/Re-exports
The following modules need re-exports in `gr_solver/__init__.py`:
- `gr_constraints` - ✅ Exists in `src.core`
- `gr_geometry` - ✅ Exists in `src.core`
- `gr_stepper` - ✅ Exists in `src.core`
- `gr_clock` - ✅ Exists in `src.core`
- `gr_solver` - ✅ Exists in `src.core`
- `host_api` - ⚠️ Only in `gr_solver/`

### 3. Broken Imports in Tests
86 import statements need verification:
- `from gr_solver.gr_solver import GRSolver` - ✅ Should work via `__init__.py`
- `from gr_solver.gr_core_fields import ...` - ✅ Should work via `__init__.py`
- `from gr_solver.gr_geometry import ...` - ✅ Should work via `__init__.py`
- `from gr_solver.phaseloom_memory import ...` - ❌ Module moved to `src/phaseloom/`
- `from gr_solver.spectral.cache import ...` - ❌ Module moved to `src/spectral/`
- `from gr_solver.elliptic.solver import ...` - ❌ Module moved to `src/elliptic/`

---

## Detailed Resolution Steps

### Phase 1: Audit and Analysis

#### Step 1.1: Verify Current Module Structure
```bash
# Check what's actually in src/core/
ls -la src/core/

# Check what gr_solver/__init__.py exports
cat gr_solver/__init__.py

# Verify symlinks
ls -la gr_solver/
```

#### Step 1.2: Identify All Broken Imports
```bash
# Find all Python files with problematic imports
grep -r "from gr_solver\." --include="*.py" . | grep -v "gr_solver/gr_solver\|gr_solver/gr_core_fields\|gr_solver/gr_geometry\|gr_solver/gr_stepper\|gr_solver/gr_clock\|gr_solver/gr_constraints"
```

#### Step 1.3: Check for Missing Module Re-exports
```bash
# Find imports that reference moved modules
grep -r "from gr_solver\.phaseloom" --include="*.py" .
grep -r "from gr_solver\.spectral" --include="*.py" .
grep -r "from gr_solver\.elliptic" --include="*.py" .
```

### Phase 2: Fix Duplicate Files

#### Step 2.1: Remove Duplicate `gr_solver/gr_core_fields.py`
```bash
# Compare files to ensure they're identical
diff gr_solver/gr_core_fields.py src/core/gr_core_fields.py

# If identical, remove the duplicate
rm gr_solver/gr_core_fields.py

# Update gr_solver/__init__.py to remove duplicate re-export
# (The __init__.py already re-exports from src.core, so this should be clean)
```

#### Step 2.2: Migrate `gr_solver/host_api.py` to `src/`
```bash
# Move the file
mv gr_solver/host_api.py src/

# Update imports in files that reference it
```

### Phase 3: Fix Missing Module Re-exports

#### Step 3.1: Update `gr_solver/__init__.py`
The current `__init__.py` only re-exports main modules. We need to add all modules:

```python
# gr_solver/__init__.py - Updated
from src.core.gr_solver import GRSolver
from src.core.gr_stepper import GRStepper
from src.core.gr_constraints import GRConstraints
from src.core.gr_geometry import GRGeometry
from src.core.gr_gauge import GRGauge
from src.core.gr_scheduler import GRScheduler
from src.core.gr_ledger import GRLedger
from src.core.gr_clock import GRClock, UnifiedClockState
from src.core.gr_clocks import MultiRateClockSystem
from src.core.gr_core_fields import GRCoreFields, SYM6_IDX, aligned_zeros
from src.core.gr_rhs import GRRhs
from src.core.gr_loc import GRLoC
from src.core.gr_sem import GRSEM
from src.core.gr_gates import GateChecker, GateKind
from src.core.gr_ttl_calculator import TTLCalculator, AdaptiveTTLs

# Re-export moved modules for backward compatibility
from src.phaseloom.phaseloom_memory import PhaseLoomMemory
from src.spectral.cache import SpectralCache
from src.elliptic.solver import EllipticSolver

# Host API
from src.host_api import GRHostAPI
```

### Phase 4: Fix Broken Test Imports

#### Step 4.1: Fix `phaseloom_memory` Imports
Files that need updates:
- `tests/test_multi_rate_clocks.py` (lines 211, 226, 243, 271, 298, 317, 346, 347)
- `tests/wp1_limit_aware_stepper_smoke.py` (lines 20)

**Current:** `from gr_solver.phaseloom_memory import PhaseLoomMemory`
**Fix:** `from src.phaseloom.phaseloom_memory import PhaseLoomMemory`

#### Step 4.2: Fix `spectral/cache` Imports
Files that need updates:
- `tests/test_gcat1_hpc_integration.py` (lines 23, 123)

**Current:** `from gr_solver.spectral.cache import SpectralCache`
**Fix:** `from src.spectral.cache import SpectralCache`

#### Step 4.3: Fix `elliptic/solver` Imports
Files that need updates:
- `tests/test_gcat1_hpc_integration.py` (line 85)

**Current:** `from gr_solver.elliptic.solver import EllipticSolver`
**Fix:** `from src.elliptic.solver import EllipticSolver`

#### Step 4.4: Fix `phaseloom_rails_gr` Imports
Files that need updates:
- `tests/wp1_limit_aware_stepper_smoke.py` (line 18)

**Current:** `from gr_solver.phaseloom_rails_gr import GRPhaseLoomRails`
**Fix:** `from src.phaseloom.phaseloom_rails_gr import GRPhaseLoomRails`

### Phase 5: Create Migration Script

#### Step 5.1: Generate Automated Import Fixer
```python
#!/usr/bin/env python3
"""
Import Migration Script
Automatically updates import statements to use new module paths.
"""

import re
import os
from pathlib import Path

# Mapping of old imports to new imports
IMPORT_MAPPINGS = {
    # Core modules (already working via __init__.py)
    "from gr_solver.gr_solver import": "from src.core.gr_solver import",
    "from gr_solver.gr_stepper import": "from src.core.gr_stepper import",
    "from gr_solver.gr_constraints import": "from src.core.gr_constraints import",
    "from gr_solver.gr_geometry import": "from src.core.gr_geometry import",
    "from gr_solver.gr_gauge import": "from src.core.gr_gauge import",
    "from gr_solver.gr_scheduler import": "from src.core.gr_scheduler import",
    "from gr_solver.gr_ledger import": "from src.core.gr_ledger import",
    "from gr_solver.gr_clock import": "from src.core.gr_clock import",
    "from gr_solver.gr_clocks import": "from src.core.gr_clocks import",
    "from gr_solver.gr_core_fields import": "from src.core.gr_core_fields import",
    "from gr_solver.gr_rhs import": "from src.core.gr_rhs import",
    "from gr_solver.gr_loc import": "from src.core.gr_loc import",
    "from gr_solver.gr_sem import": "from src.core.gr_sem import",
    "from gr_solver.gr_gates import": "from src.core.gr_gates import",
    "from gr_solver.gr_ttl_calculator import": "from src.core.gr_ttl_calculator import",
    
    # Moved modules (need full path)
    "from gr_solver.phaseloom_memory import": "from src.phaseloom.phaseloom_memory import",
    "from gr_solver.phaseloom_rails_gr import": "from src.phaseloom.phaseloom_rails_gr import",
    "from gr_solver.spectral.cache import": "from src.spectral.cache import",
    "from gr_solver.elliptic.solver import": "from src.elliptic.solver import",
    
    # Host API
    "from gr_solver.host_api import": "from src.host_api import",
}

def migrate_imports(file_path: Path) -> bool:
    """Migrate import statements in a single file."""
    content = file_path.read_text()
    original = content
    
    for old_import, new_import in IMPORT_MAPPINGS.items():
        content = content.replace(old_import, new_import)
    
    if content != original:
        file_path.write_text(content)
        return True
    return False

def main():
    """Main migration function."""
    for py_file in Path(".").rglob("*.py"):
        if py_file.parts[0] in [".git", ".pytest_cache", "__pycache__"]:
            continue
        if migrate_imports(py_file):
            print(f"Updated: {py_file}")

if __name__ == "__main__":
    main()
```

### Phase 6: Verification and Testing

#### Step 6.1: Run Import Verification
```bash
# Test imports work correctly
python -c "
from gr_solver.gr_solver import GRSolver
from gr_solver.gr_core_fields import GRCoreFields, SYM6_IDX
from gr_solver.gr_geometry import GRGeometry
from gr_solver.gr_constraints import GRConstraints
from src.phaseloom.phaseloom_memory import PhaseLoomMemory
from src.spectral.cache import SpectralCache
print('All imports successful!')
"
```

#### Step 6.2: Run Test Suite
```bash
# Run a subset of tests to verify fixes
pytest tests/test_gcat_gr_0.py -v
pytest tests/test_multi_rate_clocks.py -v
pytest tests/test_gcat1_hpc_integration.py -v

# Run full test suite
pytest tests/ -v --tb=short
```

#### Step 6.3: Fix Any Remaining Issues
- Address any import errors that surface during testing
- Fix any runtime bugs introduced by the reorganization

---

## Files Requiring Manual Review

### Files with Complex Import Patterns
1. `tests/wp1_limit_aware_stepper_smoke.py` - Multiple import types
2. `tests/test_gate_recovery.py` - Imports from `aeonic_memory_contract` (may also need migration)
3. `tests/test_adaptive_ttl.py` - Imports from `aeonic_clocks` (may also need migration)

### Files with Relative Import Issues
1. `tests/scripts/test_analysis_mode.py` - Check import paths
2. `tests/scripts/test_hadamard_minkowski.py` - Check `nsc_runtime_min` import
3. `tests/scripts/test_python_baseline.py` - Check all imports

---

## Rollback Plan

If issues arise during migration:

```bash
# Restore from git
git checkout -- gr_solver/ src/

# Re-run migration if needed
```

---

## Success Criteria

- ✅ All 86 import statements resolve without errors
- ✅ Test suite passes with >90% success rate
- ✅ No duplicate files in `gr_solver/`
- ✅ Backward compatibility maintained via `gr_solver/__init__.py`
- ✅ All moved modules accessible via new paths

---

## Next Steps

1. **Approve this plan** - Review and confirm the migration strategy
2. **Execute Phase 1** - Run audit and verification scripts
3. **Execute Phase 2** - Remove duplicate files
4. **Execute Phase 3** - Update re-exports
5. **Execute Phase 4** - Fix test imports (manual + automated)
6. **Execute Phase 5** - Run verification tests
7. **Execute Phase 6** - Full test suite run
