# Contributing to CBTSV1

Thank you for your interest in contributing to CBTSV1! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Dependencies: `pip install -e .[dev]`

### Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/cbtsv1.git
   cd cbtsv1
   ```
3. Install dependencies:
   ```bash
   pip install -e .[dev]
   ```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow the project structure conventions
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests

```bash
# Run unit tests (fast)
pytest tests/ -v --ignore=tests/test_full_stack_integration.py -m "not slow"

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### 4. Run Linting

```bash
# Check with ruff
ruff check src/ tests/

# Auto-fix
ruff check --fix src/ tests/
```

### 5. Commit Changes

Follow conventional commits:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `refactor:` for code refactoring
- `test:` for tests

### 6. Submit a Pull Request

## Code Style

- Follow PEP 8 (enforced by ruff)
- Use type hints for function signatures
- Write docstrings for public functions
- Keep line length under 100 characters

## Testing

- Unit tests: `tests/test_*.py` (fast, no external dependencies)
- Integration tests: `tests/test_*_integration.py` (slower, may need resources)
- Mark slow tests with `@pytest.mark.slow`

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new modules and functions
- Update specs in `specifications/` if contract changes

## Reporting Issues

- Use GitHub Issues
- Include reproduction steps
- Attach relevant logs and error messages
- Tag appropriately (bug, feature, enhancement)

## Code Review

- All PRs require review before merging
- Address reviewer feedback
- Keep PRs focused and small

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
