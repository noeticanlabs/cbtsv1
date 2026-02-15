# DEPRECATED: src/core/ Directory

**Status: DEPRECATED as of 2026-02-15**

This directory (`src/core/`) contains legacy GR solver implementations. The canonical location for GR solver code is now:

```
src/cbtsv1/solvers/gr/
```

## Migration Guide

All imports should be updated from:
- `from src.core.gr_solver import GRSolver` → `from cbtsv1.solvers.gr.gr_solver import GRSolver`
- `from src.core.gr_stepper import GRStepper` → `from cbtsv1.solvers.gr.gr_stepper import GRStepper`
- etc.

## Backward Compatibility

The `gr_solver/` top-level package provides backward-compatible re-exports:
```python
from gr_solver import GRSolver  # Still works, but deprecated
```

This backward compatibility will be removed in a future version.

## Why This Change?

1. **Single source of truth**: Having duplicate GR implementations in both `src/core/` and `src/cbtsv1/solvers/gr/` led to drift and confusion.

2. **Proper packaging**: The canonical location follows proper Python packaging conventions with the `cbtsv1` package namespace.

3. **Clear ownership**: GR solver code is now clearly part of the `cbtsv1.solvers` subsystem.

## Timeline

- **v0.1.x**: Deprecation warning, backward compatibility maintained
- **v0.2.x**: Remove backward compatibility for direct imports from `src.core`
- **v1.0**: Remove `src/core/` directory entirely
