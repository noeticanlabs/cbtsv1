# Modularization Master Plan

## Overview

This plan documents the systematic modularization of 19 large Python modules (>700 lines) in the project. The goal is to refactor each module into smaller, focused components while maintaining functionality and ensuring no imports or dependencies are broken.

## Modules Identified (>700 Lines)

### NSC/Compiler Modules (Core)
| File | Lines | Refactoring Priority |
|------|-------|---------------------|
| `src/nsc/parse.py` | 2029 | 1 (Highest - most complex) |
| `src/nllc/parse.py` | 1444 | 2 |
| `src/nsc/exec_vm.py` | 1249 | 3 |
| `src/nsc/ast.py` | 1046 | 4 (Foundation - be careful) |
| `src/nsc/type_checker.py` | 935 | 5 |
| `src/nsc/disc_lower.py` | 742 | 6 |

### GR Solver Modules (Core)
| File | Lines | Refactoring Priority |
|------|-------|---------------------|
| `src/core/gr_solver.py` | 1675 | 1 (Highest) |
| `src/core/gr_rhs.py` | 1084 | 2 |
| `src/core/gr_stepper.py` | 914 | 3 |
| `src/core/gr_geometry.py` | 861 | 4 |

### NLLC Modules
| File | Lines | Refactoring Priority |
|------|-------|---------------------|
| `src/nllc/ast.py` | 815 | 1 (Foundation) |

### Test Modules
| File | Lines | Refactoring Priority |
|------|-------|---------------------|
| `tests/test_nsc_enhanced_geometry.py` | 1193 | Low (Tests only) |
| `tests/test_nsc_math_physics_accuracy.py` | 1164 | Low (Tests only) |
| `tests/test_gcat1_suite.py` | 929 | Low (Tests only) |
| `tests/test_comprehensive_gr_solver.py` | 888 | Low (Tests only) |
| `tests/test_nsc_regression.py` | 797 | Low (Tests only) |
| `tests/test_1_mms_lite.py` | 729 | Low (Tests only) |
| `tests/test_mms_convergence.py` | 709 | Low (Tests only) |

### Configuration Files
| File | Lines | Refactoring Priority |
|------|-------|---------------------|
| `config/gr_rhs.nsc` | 718 | Medium (Configuration) |

## Dependency Analysis

### Dependency Graph (Key Modules Only)

```
src/nsc/ast.py (1046)
    └── imports: dataclasses, typing, enum
    └── used by: parse.py, type_checker.py, exec_vm.py, disc_lower.py

src/nllc/ast.py (815)
    └── imports: dataclasses, typing
    └── used by: nllc/parse.py, nllc/exec_vm.py

src/nsc/parse.py (2029)
    └── imports: .ast, .lex
    └── used by: test_nsc_parser.py, m3l_compiler.py
    └── exports: parse_program, parse_string, Parser class

src/nllc/parse.py (1444)
    └── imports: .ast, .lex
    └── used by: test_nllc_parser.py, nllc/exec_vm.py
    └── exports: parse_program, parse_string, Parser class

src/nsc/type_checker.py (935)
    └── imports: .ast
    └── used by: nsc/pipeline.py, m3l_compiler.py

src/nsc/disc_lower.py (742)
    └── imports: .ast
    └── used by: nsc/pipeline.py, m3l_compiler.py

src/nsc/exec_vm.py (1249)
    └── imports: .ast, .type_checker
    └── used by: nsc/pipeline.py, m3l_compiler.py

src/core/gr_solver.py (1675)
    └── imports: gr_geometry, gr_constraints, gr_stepper, gr_scheduler, gr_ledger
    └── imports: gr_gauge, phaseloom_gr_orchestrator, aeonic_memory_bank
    └── exports: GRSolver class

src/core/gr_rhs.py (1084)
    └── imports: gr_geometry, gr_constraints
    └── used by: gr_solver.py, gr_stepper.py

src/core/gr_stepper.py (914)
    └── imports: gr_constraints, gr_rhs, gr_scheduler, gr_ledger, gr_gates
    └── used by: gr_solver.py

src/core/gr_geometry.py (861)
    └── imports: gr_core_fields
    └── used by: gr_solver.py, gr_rhs.py, gr_constraints.py, gr_stepper.py
```

## Safe Refactoring Order

### Phase 1: Foundation Modules (Low Risk)
1. `src/nsc/ast.py` (1046) - AST definitions (ALL other NSC modules depend on this)
2. `src/nllc/ast.py` (815) - AST definitions (ALL other NLLC modules depend on this)

### Phase 2: Parser Modules (Medium Risk)
1. `src/nsc/parse.py` (2029) - Parse into: `statement_parser.py`, `expr_parser.py`, `type_parser.py`
2. `src/nllc/parse.py` (1444) - Parse into: `nllc_statement_parser.py`, `nllc_expr_parser.py`

### Phase 3: Type System & Execution (Medium Risk)
1. `src/nsc/type_checker.py` (935) - May need to split by type category
2. `src/nsc/disc_lower.py` (742) - Split by discretization type
3. `src/nsc/exec_vm.py` (1249) - Split into: `vm_core.py`, `vm_ops.py`, `vm_interpreter.py`

### Phase 4: GR Solver Modules (High Risk - Core Functionality)
1. `src/core/gr_geometry.py` (861) - Split into: `christoffel_ops.py`, `ricci_ops.py`, `curvature_ops.py`
2. `src/core/gr_rhs.py` (1084) - Split by RHS category
3. `src/core/gr_stepper.py` (914) - Split into: `rk4_integrator.py`, `gate_checker.py`, `constraint_damping.py`
4. `src/core/gr_solver.py` (1675) - LAST - This orchestrates everything

### Phase 5: Test Modules (Low Risk)
These can be refactored anytime, but prioritize by frequency of use:
1. `tests/test_nsc_enhanced_geometry.py`
2. `tests/test_gcat1_suite.py`
3. `tests/test_comprehensive_gr_solver.py`

## Safety Checklist for Each Module Refactoring

### Pre-Refactoring Checklist
- [ ] Run all existing tests to establish baseline
- [ ] Document all public API (functions, classes, constants)
- [ ] Map all import statements (what uses this module)
- [ ] Identify circular dependencies
- [ ] Create dependency impact analysis
- [ ] Set up import tracing script

### During Refactoring Checklist
- [ ] Create new module file with extracted code
- [ ] Maintain exact function signatures
- [ ] Preserve docstrings
- [ ] Keep all imports in new module
- [ ] Update original module to import from new module (re-export)
- [ ] Run import checks after each extraction

### Post-Refactoring Checklist
- [ ] Run all tests - verify no regressions
- [ ] Check import paths using static analysis
- [ ] Verify circular imports are resolved
- [ ] Run mypy/pyright type checking
- [ ] Check code coverage remains stable
- [ ] Update any documentation referencing old structure
- [ ] Commit changes with descriptive message

## Rollback Procedures

### Git-Based Rollback
```bash
# If something breaks, rollback the last commit
git revert HEAD
git log --oneline -5
```

### Module-Level Rollback
1. Keep original module file with extracted code commented out
2. Restore imports in dependent modules temporarily
3. Run tests to confirm rollback success
4. Remove commented code after successful rollback

## Import Verification Strategy

### Automated Import Checker Script
```python
# scripts/verify_imports.py
"""
Script to verify all imports are valid after refactoring.
"""
import ast
import sys
from pathlib import Path

def check_module_imports(module_path: Path) -> list:
    """Check all imports in a module are valid."""
    # Implementation details...
    pass

def find_broken_imports(project_root: Path) -> dict:
    """Find all broken imports in the project."""
    # Implementation details...
    pass
```

### Key Verification Steps
1. Run `python -c "import src.nsc.parse; print('OK')"` on each refactored module
2. Check `pytest` suite passes
3. Verify `mypy` type checking passes
4. Run `scripts/update_imports.py` if available

## Module Extraction Patterns

### Pattern 1: Class Extraction
```python
# Before: All in one class
class LargeParser:
    def parse_statement(self): ...
    def parse_expr(self): ...
    def parse_type(self): ...
    def parse_directive(self): ...

# After: Extract to separate files
from .statement_parser import StatementParser
from .expr_parser import ExprParser
from .type_parser import TypeParser
from .directive_parser import DirectiveParser

class LargeParser(StatementParser, ExprParser, TypeParser, DirectiveParser):
    pass
```

### Pattern 2: Function Group Extraction
```python
# Before: Mixed function groups
def parse_stmt(self): ...
def parse_decl(self): ...
def parse_equation(self): ...

def type_ops(self): ...
def type_check(self): ...
def type_infer(self): ...

# After: Separate modules
from . import stmt_parsing
from . import type_operations

class Parser:
    parse_stmt = stmt_parsing.parse_stmt
    parse_decl = stmt_parsing.parse_decl
    type_check = type_operations.type_check
```

### Pattern 3: Constant/Config Extraction
```python
# Before: Scattered constants
class Parser:
    ERROR_EOF = 6
    ERROR_UNEXPECTED = 7
    
    def __init__(self):
        self.error_map = {...}

# After: Constants module
from .parser_constants import ParserConstants, ErrorMessages

class Parser:
    ERROR_EOF = ParserConstants.ERROR_EOF
```

## Test Coverage Requirements

### For Each Module Refactoring
1. Unit tests for extracted module should exist
2. Integration tests should pass
3. Import tests should verify module loads
4. Type checking should pass (if using type hints)

### Test Files to Update
- `tests/test_nsc_parser.py` - For parse.py changes
- `tests/test_nsc_type_checker.py` - For type_checker.py changes
- `tests/test_gr_components.py` - For GR solver changes
- `tests/test_comprehensive_gr_solver.py` - For full integration tests

## Migration Scripts

### For Each Module Phase
```bash
# Phase 1: Extract AST definitions
python scripts/extract_ast_definitions.py

# Phase 2: Extract parser components
python scripts/extract_parser_components.py --module nsc.parse --output src/nsc/parsers/

# Phase 3: Extract GR solver components
python scripts/extract_gr_components.py --module core.gr_solver --output src/core/solver/
```

## Timeline & Execution

### Recommended Approach
1. **One module at a time** - Complete refactoring before moving to next
2. **Full test suite passes** - Before starting next module
3. **Commit after each module** - Atomic commits for rollback
4. **Document changes** - Update this plan with lessons learned

### Expected Duration (Per Module)
- Small modules (<800 lines): 1-2 hours
- Medium modules (800-1200 lines): 2-4 hours
- Large modules (1200+ lines): 4-8 hours

## Success Criteria

1. **No import errors** - All modules import successfully
2. **All tests pass** - pytest suite passes 100%
3. **Type checking passes** - mypy/pyright reports no errors
4. **Code coverage maintained** - Coverage doesn't decrease
5. **Documentation updated** - README and docs reflect new structure

## Risk Mitigation

### High-Risk Modules
- `src/core/gr_solver.py` - Test thoroughly, may need rollback
- `src/nsc/parse.py` - Many dependent modules

### Mitigation Strategies
1. Extensive testing before and after
2. Incremental extraction (not one-shot)
3. Keep original file with re-exports during transition
4. Use feature flags for gradual rollout

## Next Steps

1. Review and approve this plan
2. Create extraction scripts for Phase 1 modules
3. Begin with `src/nsc/ast.py` extraction
4. Update this document as lessons are learned

---
Plan Version: 1.0
Created: 2024
Status: Draft - Awaiting Approval
