# CBTSV1 - General Relativity Solver Project

A 3+1 General Relativity solver with PhaseLoom coherence framework, AEONIC memory system, and NSC compiler infrastructure.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/noeticanlabs/cbtsv1.git
cd cbtsv1

# Install dependencies
pip install -e .[dev]

# Verify installation
pytest --version
ruff --version
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run unit tests only (fast)
pytest tests/ -v -m "not slow" --ignore=tests/test_full_stack_integration.py

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_mms_convergence.py -v
```

## Project Structure

```
cbtsv1/
├── src/                          # Primary source code
│   ├── aeonic/                   # AEONIC memory system
│   ├── common/                   # Shared utilities
│   ├── core/                     # Core GR solver (20+ modules)
│   ├── contracts/                # Contract implementations
│   ├── elliptic/                 # Elliptic solvers
│   ├── hadamard/                 # Hadamard VM
│   ├── module/                   # Module system
│   ├── phaseloom/                # Phaseloom integration
│   ├── solver/                   # PIR representation
│   ├── spectral/                 # Spectral methods
│   └── triaxis/                  # Triaxis lexicon
├── tests/                        # Test suite
├── config/                       # NSC configuration files
├── data/                         # Test data and receipts
├── docs/                         # Documentation
├── specifications/               # Technical specifications
│   ├── glyphs/                   # Glyph codebooks
│   ├── lexicon/                  # Lexicon specs
│   ├── contracts/                # Contract specs
│   └── theory/                   # Mathematical theory
├── scripts/                      # Utility scripts
├── plans/                        # Planning documents
└── Technical_Data/               # Research documents
```

## Key Components

### Core Solver ([`src/core/`](src/core))

| Module | Purpose |
|--------|---------|
| `gr_solver.py` | Main GR solver |
| `gr_stepper.py` | Time stepper with coherence gates |
| `gr_constraints.py` | Constraint handling (Hamiltonian/Momentum) |
| `gr_geometry.py` | Geometry operations (Christoffel, Ricci) |
| `gr_coherence.py` | Coherence operator with λ-damping |
| `gr_clock.py` | Temporal coherence tracking |

### Contracts ([`src/contracts/`](src/contracts))

- `stepper_contract.py` - Step acceptance criteria
- `phaseloom_contract.py` - Band coherence governance
- `omega_ledger.py` - Immutable receipt chain

### PhaseLoom ([`src/phaseloom/`](src/phaseloom))

Multi-thread governance system with:
- Dyadic band analysis
- Kuramoto-style coherence order parameter
- Window-level regime classification

## Documentation

- **[Coherence Thesis](Technical_Data/coherence_thesis_extended_canon_v2_1.md)** - Comprehensive coherence framework
- **[Plans](plans/)** - Development roadmaps
- **[Specifications](specifications/)** - Technical specs and contracts

## Development

### Code Style

- Python 3.10+
- Follow PEP 8 (enforced by ruff)
- Use type hints
- Write docstrings

### Linting

```bash
# Check code
ruff check src/ tests/

# Auto-fix
ruff check --fix src/ tests/
```

### CI/CD

GitHub Actions runs on every PR:
- Unit tests (Python 3.10, 3.11)
- Linting with ruff
- Coverage reporting
- Syntax validation

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE)

## References

- Clay Mathematics Institute Problems (Yang-Mills, Navier-Stokes)
- Numerical Relativity: Computational Methods
- Constraint Damping in Generalized Harmonic Gauge
