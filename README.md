# CBTSV1 - General Relativity Solver Project

## Project Structure

```
cbtsv1/
├── src/                          # Primary source code
│   ├── aeonic/                   # AEONIC memory system
│   ├── aml/                      # AML utilities
│   ├── common/                   # Shared utilities
│   ├── core/                     # Core GR solver (20+ modules)
│   ├── contracts/                # Contract implementations
│   ├── elliptic/                 # Elliptic solvers
│   ├── hadamard/                 # Hadamard VM
│   ├── module/                   # Module system
│   ├── nllc/                     # NIR/LLVM compiler
│   ├── nsc/                      # Numerical solver compiler
│   ├── phaseloom/                # Phaseloom integration
│   ├── receipts/                 # Receipt system
│   ├── solver/                   # PIR representation
│   ├── spectral/                 # Spectral methods
│   ├── tgs/                      # TGS design
│   └── triaxis/                  # Triaxis lexicon
├── tests/                        # Test suite
│   └── scripts/                  # Test scripts
├── config/                       # NSC configuration files
├── data/                         # Test data and receipts
│   ├── receipts/                 # Receipt JSON files
│   └── test_data/                # Test data files
├── docs/                         # Documentation
├── specifications/               # Technical specifications
│   ├── glyphs/                   # Glyph codebooks and taxonomies
│   ├── lexicon/                  # Lexicon specifications
│   ├── praxica/                  # Praxica specifications
│   ├── aeonica/                  # Aeonica specifications
│   ├── contracts/                # Contract specifications
│   └── theory/                   # Mathematical theory documents
├── scripts/                      # Utility scripts
├── plans/                        # Planning documents
└── gr_solver/                    # Symlink to src/core (backward compatibility)
```

## Key Directories

### [`src/core/`](src/core)
Core GR solver modules including:
- `gr_solver.py` - Main solver
- `gr_stepper.py` - Time stepper
- `gr_constraints.py` - Constraint handling
- `gr_geometry.py` - Geometry operations
- `gr_clock.py`, `gr_clocks.py` - Clock systems
- `gr_rhs.py` - Right-hand side computations

### [`src/contracts/`](src/contracts)
Contract implementations:
- `solver_contract.py`
- `stepper_contract.py`
- `phaseloom_contract.py`
- `orchestrator_contract.py`
- `temporal_system_contract.py`
- `omega_ledger.py`

### [`src/nllc/`](src/nllc)
NIR/LLVM compiler for the solver:
- `nir.py` - NIR representation
- `lower_nir.py` - Lowering to LLVM
- `vm.py` - Virtual machine
- `aeonic.py` - AEONIC integration

### [`specifications/`](specifications)
Technical specifications organized by domain:
- **glyphs/** - Glyph taxonomies and codebooks
- **lexicon/** - Lexicon specifications
- **praxica/** - Praxica specifications
- **aeonica/** - Aeonica specifications
- **contracts/** - Contract specifications
- **theory/** - Mathematical theory documents

### [`data/`](data)
- **receipts/** - Receipt JSONL files for test verification
- **test_data/** - Test configuration files (E1, E2 series)

### [`scripts/`](scripts)
Utility scripts for:
- Compilation (`compile_nllc.py`, `nsc_compile_min.py`)
- Diagnostics (`diagnostic_*.py`)
- Testing (`dt_sweep.py`, `phaseloom_27.py`)

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_mms_convergence.py

# Run with coverage
pytest --cov=src tests/
```

## Configuration

NSC configuration files are located in [`config/`](config):
- `gr_gate_policy.nscb`
- `policy_bundle.nscb`
- `coupling_policy_v0.1.json`
- `gr_constraints_nsc.py`
- `gr_rhs.nscir.json`

## Development

This project uses a structured layout with clear module boundaries. When adding new code:

1. Place core modules in `src/core/`
2. Place compiler-related code in `src/nllc/` or `src/nsc/`
3. Place integration code in `src/phaseloom/`
4. Place tests in `tests/scripts/` or categorize appropriately
5. Place specifications in `specifications/` subdirectories

## Backward Compatibility

The `gr_solver/` directory is a symlink to `src/core/` for backward compatibility with existing imports:
```bash
gr_solver/ -> src/core/
```

## Removed Directories

The following legacy directories have been removed:
- `Technical Data/` - Consolidated into `specifications/`
- `Project Data/` - Empty directory, removed
- `gr_gate_policy_dir/` - Consolidated into `config/`
- `noetica_nsc_phase1/` - Legacy, removed

## Python Path Setup

For development, ensure `src/` is in your Python path:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

Or add to `pyproject.toml`:
```toml
[tool.poetry.packages]
include = ["src/*"]
```
