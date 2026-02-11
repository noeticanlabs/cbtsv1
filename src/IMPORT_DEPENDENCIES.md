# Import Dependencies Documentation

This document specifies the import order and circular dependency constraints for the GR solver system.

## Safe Import Order

The following import chain is verified to be safe (no circular dependencies):

```
gr_gates → gr_constraints → gr_core_fields
gr_core_fields → gr_clock → gr_scheduler
gr_scheduler → gr_clocks → MultiRateBandManager
phaseloom_rails_gr → gr_gates, gr_core_fields
```

## Module Dependency Graph

### Core Module Hierarchy

1. **gr_core_fields**: Fundamental field utilities (no external gr dependencies)
   - Utilities: `inv_sym6`, `trace_sym6`, `norm2_sym6`, `det_sym6`, `eigenvalues_sym6`, `cond_sym6`
   - Safe to import from: `gr_gates`, `gr_constraints`, `gr_scheduler`, `phaseloom_rails_gr`

2. **gr_gates**: Gate checking and NSC integration
   - Imports from: `gr_core_fields`
   - Safe to import from: `gr_constraints`, `phaseloom_rails_gr`

3. **gr_constraints**: Constraint computation and elliptic cleanup
   - Imports from: `gr_core_fields`, `elliptic.solver`
   - Safe to import from: `gr_scheduler`

4. **gr_clock**: Unified clock state management
   - Imports from: (minimal external dependencies)
   - Safe to import from: `gr_scheduler`, `gr_clocks`, `phaseloom_memory`

5. **gr_scheduler**: Timestep selection and clock coordination
   - Imports from: `gr_clock`
   - Safe to import from: `gr_clocks`, `phaseloom_memory`

6. **gr_clocks**: Multi-rate clock system
   - Imports from: `gr_clock`, `gr_scheduler`
   - Safe to import from: `phaseloom_memory`

### Application Modules

- **phaseloom_rails_gr**: Rail system for PhaseLoom
  - Imports from: `gr_gates`, `gr_core_fields` (moved to top level, see below)
  - No circular dependencies

- **phaseloom_memory**: Memory tracking with clock integration
  - Imports from: `gr_clocks`, `gr_clock`
  - No circular dependencies

## Late Import Resolution

All late imports have been resolved by moving them to module top level:

### phaseloom_rails_gr.py

**Before:**
```python
def _get_gamma_diagnostics(self, fields):
    # ...
    from src.core.gr_core_fields import det_sym6, eigenvalues_sym6, cond_sym6
    # ...

def __init__(self, ...):
    # ...
    from src.core.gr_gates import GateChecker
    # ...
```

**After:**
```python
# At module top (lines 15-18)
import numpy as np
import json
from src.core.gr_gates import GateChecker
from src.core.gr_core_fields import det_sym6, eigenvalues_sym6, cond_sym6, inv_sym6

# Methods use top-level imports directly
```

## Circular Dependency Prevention Rules

**SAFE:**
- ✓ gr_gates → gr_core_fields
- ✓ gr_constraints → gr_core_fields
- ✓ gr_scheduler → gr_clock
- ✓ gr_clocks → gr_scheduler
- ✓ phaseloom_rails_gr → gr_gates, gr_core_fields

**UNSAFE (Never use):**
- ✗ gr_gates ← gr_constraints (would create cycle)
- ✗ gr_clock ← gr_scheduler (would create cycle)
- ✗ gr_core_fields ← (any module except as TYPE_CHECKING)

## TYPE_CHECKING Pattern for Forward References

For type hints that would cause circular imports, use:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.gr_gates import GateChecker  # Type hints only
    from src.core.gr_scheduler import GRScheduler

class MyClass:
    def set_gate_checker(self, checker: 'GateChecker') -> None:  # String quotes for forward ref
        self.gate_checker = checker
```

## Import Order Enforcement

To verify no new circular imports are introduced:

1. Run circular import check:
   ```bash
   python -c "import src.core.gr_gates; import src.core.gr_constraints"
   ```

2. Check specific module:
   ```bash
   python -c "import src.phaseloom.phaseloom_rails_gr"
   ```

## Changes Made (Phase 3)

- **gr_scheduler.py**: Lines 16 (import UnifiedClock) - already at top
- **gr_clocks.py**: Lines 4 (import statements) - already at top
- **phaseloom_rails_gr.py**: 
  - Lines 18-19: Moved GateChecker and gr_core_fields imports to top
  - Lines 34, 64: Removed late imports, use top-level instead
- **nsc_enhanced_solvers.py**: Lines 15-17 (imports) - already at top
- **nsc_compact_fd_stabilized.py**: Lines 17-19 (imports) - already at top
- **gr_geometry.py**: Lines 15-16 (imports) - already at top

All modules now have explicit imports at module level with documented safe import order.
