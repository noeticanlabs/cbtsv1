# Project Reorganization Plan

## Current State Analysis

### Root Directory Issues (~80+ files at root level)
The project has significant root directory clutter with mixed file types:
- **Python files**: `aeonic_*.py`, `aml.py`, `diagnostic_*.py`, `test_*.py`, etc.
- **JSON/NDJSON**: `diagnostic_*.json`, `receipts_*.json`, `test_results_*.json`
- **Compiler files**: `.nsc`, `.nscb`, `.nscir.json`
- **Markdown plans**: `Phase*_*.md`, build plans
- **Standalone scripts**: `phaseloom_27.py`, `promotion_engine.py`, etc.

### Structural Issues Identified

| Issue | Location | Impact |
|-------|----------|--------|
| Duplicate module directories | `gr_solver/` + `src/` | Confusion on which is authoritative |
| Test files scattered | Root + `tests/` | Hard to discover and run |
| Config files mixed with code | Root level | Pollutes module namespace |
| Technical Data not organized | `Technical Data/` | Specs mixed with data sheets |
| Empty directory | `Project Data/` | Should be removed or populated |
| Phase plans in root | `Phase*_*.md` | Should be in `plans/` |

### Module Analysis

#### `src/` Directory (Well-organized)
```
src/
├── common/         ✓ Reusable utilities
├── hadamard/       ✓ Dedicated compiler module
├── module/         ✓ Manifest system
├── nllc/           ✓ NIR/LLVM compiler
├── nsc/            ✓ Numerical solver compiler
├── solver/         ✓ PIR representation
└── triaxis/        ✓ Lexicon system
```

#### `gr_solver/` Directory (Needs cleanup)
```
gr_solver/
├── Core modules (gr_*.py)         ~20 files
├── Contract files (*_contract.py) ~10 files
├── Phaseloom adapters             ~10 files
├── Elliptic subdirectory          1 file
└── Spectral subdirectory          1 file
```

#### `tests/` Directory (Needs better organization)
```
tests/
├── test_*.py              General tests
├── receipts_*.jsonl       Receipt data
└── generate_*.py          Test generators
```

---

## Target Structure

```
cbtsv1/
├── src/                          # Primary source code
│   ├── common/                   # Shared utilities
│   │   ├── __init__.py
│   │   ├── bundle.py
│   │   └── receipt.py
│   ├── core/                     # Core GR solver (migrated from gr_solver/)
│   │   ├── __init__.py
│   │   ├── clock.py
│   │   ├── coherence.py
│   │   ├── constraints.py
│   │   ├── geometry.py
│   │   ├── gauge.py
│   │   ├── gates.py
│   │   ├── ledger.py
│   │   ├── loc.py
│   │   ├── receipts.py
│   │   ├── rhs.py
│   │   ├── scheduler.py
│   │   ├── sem.py
│   │   ├── solver.py
│   │   ├── stepper.py
│   │   └── ttl_calculator.py
│   ├── contracts/                # Contract implementations
│   │   ├── __init__.py
│   │   ├── orchestrator.py
│   │   ├── phaseloom.py
│   │   ├── solver.py
│   │   ├── stepper.py
│   │   ├── temporal_system.py
│   │   └── omega_ledger.py
│   ├── phaseloom/                # Phaseloom integration
│   │   ├── __init__.py
│   │   ├── adapter.py
│   │   ├── controller.py
│   │   ├── memory.py
│   │   ├── octaves.py
│   │   ├── orchestrator.py
│   │   ├── rails.py
│   │   ├── receipts.py
│   │   ├── render.py
│   │   └── threads.py
│   ├── elliptic/                 # Elliptic solvers
│   │   ├── __init__.py
│   │   └── solver.py
│   ├── spectral/                 # Spectral methods
│   │   ├── __init__.py
│   │   └── cache.py
│   ├── hadamard/                 # Hadamard VM
│   │   ├── __init__.py
│   │   ├── assembler.py
│   │   ├── compiler.py
│   │   └── vm.py
│   ├── nllc/                     # NIR/LLVM compiler
│   │   ├── __init__.py
│   │   ├── aeonic.py
│   │   ├── ast.py
│   │   ├── intrinsic_binder.py
│   │   ├── lex.py
│   │   ├── lower_nir.py
│   │   ├── nir.py
│   │   ├── parse.py
│   │   └── vm.py
│   ├── nsc/                      # Numerical solver compiler
│   │   ├── __init__.py
│   │   ├── assemble_pde.py
│   │   ├── ast.py
│   │   ├── compile_bc.py
│   │   ├── flatten.py
│   │   ├── lex.py
│   │   ├── lower_pir.py
│   │   └── parse.py
│   ├── triaxis/                  # Triaxis lexicon
│   │   ├── __init__.py
│   │   └── lexicon.py
│   ├── module/                   # Module system
│   │   ├── __init__.py
│   │   └── manifest.py
│   └── host_api.py               # Host API (migrated from gr_solver/)
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/                     # Unit tests
│   │   ├── __init__.py
│   │   ├── test_mms_*.py
│   │   ├── test_coherence.py
│   │   ├── test_constraints.py
│   │   └── test_*.py
│   ├── integration/              # Integration tests
│   │   ├── __init__.py
│   │   ├── test_gcat_gr_*.py
│   │   ├── test_schwarzschild.py
│   │   └── test_*.py
│   ├── contracts/                # Contract tests
│   │   ├── __init__.py
│   │   └── test_*.py
│   ├── phaseloom/                # Phaseloom tests
│   │   ├── __init__.py
│   │   └── test_*.py
│   ├── receipts/                 # Receipt test data
│   │   └── *.jsonl
│   ├── generate_*.py             # Test generators
│   └── test_gr_adapter.py
│
├── config/                       # Configuration files
│   ├── gr_gate_policy.nscb
│   ├── policy_bundle.nscb
│   ├── golden_run.nscb
│   ├── coupling_policy_v0.1.json
│   └── receipts_schema.json
│
├── docs/                         # Documentation
│   ├── memory_system_architecture.md
│   ├── unified_clock_documentation.md
│   └── api_reference.md          # Generated from docstrings
│
├── specifications/               # Technical specifications
│   ├── glyphs/
│   │   ├── 12_GLYPH_TAXONOMY.md
│   │   ├── 14_HADAMARD_GLYPH_CODEBOOK.md
│   │   └── 14_TRIASIS_GLYPH_CODEBOOK_NPA.md
│   ├── lexicon/
│   │   ├── GHLL_LEXICON_ADDITIONS_AND_CONTRACT.md
│   │   ├── GLLL_LEXICON_ADDITIONS_AND_CONTRACT.md
│   │   ├── GML_LEXICON_ADDITIONS_AND_CONTRACT.md
│   │   └── project_lexicon_canon_v1_2.md
│   ├── praxica/
│   │   ├── 30_PRAXICA_SPEC.md
│   │   └── kr1_execution_spec.md
│   ├── aeonica/
│   │   ├── 42_AEONICA_RECEIPTS.md
│   │   └── aeonic_phaseloom_canon_spec_v1_0.md
│   ├── contracts/
│   │   ├── solver_contract.md
│   │   ├── stepper_contract.md
│   │   ├── temporal_system_contract.md
│   │   └── orchestrator_contract.md
│   └── theory/
│       ├── clay_theorem_ledger_yang_mills.md
│       ├── loc_clay_proof_skeleton_yang_mills.md
│       └── loc_clay_proof_skeleton_navier_stokes.md
│
├── data/                         # Test data and receipts
│   ├── receipts/
│   │   ├── aeonic_receipts.jsonl
│   │   ├── golden_receipts.jsonl
│   │   └── test_receipts.jsonl
│   ├── diagnostic/
│   │   ├── diagnostic_e1_analysis.json
│   │   ├── diagnostic_flight_e1.json
│   │   ├── diagnostic_stage_jumps_e1.json
│   │   └── diagnostic_time_levels.json
│   └── results/
│       ├── test_results_N8.json
│       ├── test_results_N12.json
│       └── test_results_N16.json
│
├── scripts/                      # Utility scripts
│   ├── compile_nllc.py
│   ├── debug_test.py
│   ├── dt_sweep.py
│   ├── nsc_compile_min.py
│   ├── nsc_runtime_min.py
│   ├── promotion_engine.py
│   └── timing_script.py
│
├── plans/                        # Planning documents (consolidated)
│   ├── aeonic_memory_contract_alignment_plan.md
│   ├── comprehensive_gr_solver_system_test_plan.md
│   ├── gcat2_scenario1_high_frequency_gauge_pulse.md
│   ├── gcat2_scenario2_constraint_violating_perturbation.md
   ├── gcat2_scenario3_under_resolution_cascade.md
│   ├── gr_solver_build_plan.md
│   ├── gr_solver_improvements_plan.md
│   ├── gr_solver_modularization_plan.md
│   ├── hadamard_glyph_system_spec.md
│   ├── hpc_integration_plan.md
│   ├── lexicon.py
│   ├── minkowski_stability_test_plan.md
│   ├── nsc_compilation_pipeline_design.md
│   ├── nsc_phaseloom_gr_nr_integration_plan.md
│   ├── nsc_split_plan.md
│   ├── nsc_upgrade_plan.md
│   ├── triaxis_integration_master_plan.md
│   └── Phase1_Import_Syntax_DepClosureHash_NamespaceReceipts_BuildPlan_v6_1.md
│   └── Phase2_TwinTrack_RuntimeAndSolver_BuildPlan_v0_2.md
│
├── notebooks/                    # Jupyter notebooks (optional)
│   └── .keep
│
├── .github/                      # GitHub configuration
│   └── workflows/
│       └── ci.yml
│
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Poetry/PEP 518 config
├── README.md                     # Project overview
├── .gitignore
├── .roo/                         # Roo configuration
│   └── rules/
│       └── rules.md
├── .idx/                         # Indexing (can be gitignored)
├── .pytest_cache/                # Test cache (can be gitignored)
└── gr_solver/                    # SYMLINK to src/core (backward compatibility)
```

---

## Migration Plan

### Phase 1: Create new directory structure
```bash
# Create directories
mkdir -p src/core src/contracts src/phaseloom src/elliptic src/spectral
mkdir -p config
mkdir -p specifications/glyphs specifications/lexicon specifications/praxica
mkdir -p specifications/aeonica specifications/contracts specifications/theory
mkdir -p data/receipts data/diagnostic data/results
mkdir -p scripts
mkdir -p tests/unit tests/integration tests/contracts tests/phaseloom tests/receipts
```

### Phase 2: Migrate source files
```bash
# Migrate gr_solver/ contents
mv gr_solver/gr_clock.py src/core/
mv gr_solver/gr_coherence.py src/core/
mv gr_solver/gr_constraints.py src/core/
# ... (all gr_*.py files to src/core/)
mv gr_solver/*_contract.py src/contracts/
mv gr_solver/phaseloom_*.py src/phaseloom/
mv gr_solver/elliptic/solver.py src/elliptic/
mv gr_solver/spectral/cache.py src/spectral/
mv gr_solver/host_api.py src/
```

### Phase 3: Migrate tests
```bash
# Organize tests by type
mv tests/test_1*.py tests/unit/
mv tests/test_2*.py tests/unit/
# ... (categorize by test type)
mv tests/test_*contract*.py tests/contracts/
mv tests/test_*phaseloom*.py tests/phaseloom/
```

### Phase 4: Migrate configuration files
```bash
# Move compiler configs
mv *.nscb config/
mv *policy*.json config/
mv coupling_policy*.json config/
```

### Phase 5: Migrate specifications
```bash
# Organize specs
mv Technical\ Data/*GLYPH* specifications/glyphs/
mv Technical\ Data/*LEXICON* specifications/lexicon/
mv Technical\ Data/*PRAXICA* specifications/praxica/
mv Technical\ Data/*AEONICA* specifications/aeonica/
mv Technical\ Data/*CONTRACT*.md specifications/contracts/
mv Technical\ Data/*clay* specifications/theory/
```

### Phase 6: Migrate data and scripts
```bash
# Move data files
mv *receipts*.jsonl data/receipts/
mv *diagnostic*.json data/diagnostic/
mv *results*.json data/results/

# Move utility scripts
mv compile_nllc.py scripts/
mv debug_test.py scripts/
mv dt_sweep.py scripts/
# ... (other standalone scripts)
```

### Phase 7: Consolidate plans
```bash
# Move plans from root
mv Phase*_*.md plans/
```

### Phase 8: Cleanup
```bash
# Remove empty directories
rmdir "Project Data" 2>/dev/null || true

# Create backward compatibility symlink
ln -s src/core gr_solver

# Update imports in all files (use sed or IDE refactor)
```

### Phase 9: Update Python path and imports
```python
# Add to pyproject.toml or setup.py
[tool.poetry.packages]
include = [
    "src/common",
    "src/core",
    "src/contracts",
    "src/phaseloom",
    "src/elliptic",
    "src/spectral",
    "src/hadamard",
    "src/nllc",
    "src/nsc",
    "src/triaxis",
    "src/module",
]
```

---

## Benefits of New Structure

| Category | Improvement |
|----------|-------------|
| **Discoverability** | Clear module boundaries, logical grouping |
| **Maintainability** | Single source of truth, no duplicate directories |
| **Testing** | Organized test hierarchy with unit/integration separation |
| **Documentation** | Separate specs from code, easier to navigate |
| **Configuration** | Isolated config directory, version controllable |
| **Scalability** | Clear pattern for adding new modules |
| **CI/CD** | Easier to configure build and test pipelines |
| **Onboarding** | New developers can understand structure quickly |

---

## Risk Mitigation

1. **Backward Compatibility**: Create symlink `gr_solver/ -> src/core/`
2. **Incremental Migration**: Move files module by module, test after each
3. **Import Updates**: Use IDE refactoring or automated sed scripts
4. **Git History**: Preserve history with `git mv` commands
5. **Rollback Plan**: Keep original structure in a backup branch during migration

---

## Next Steps

1. Review and approve this plan
2. Begin Phase 1: Create new directory structure
3. Execute migration in phases with testing
4. Update all import statements
5. Verify all tests pass
6. Update documentation
