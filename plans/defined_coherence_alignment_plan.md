# Defined Coherence Alignment Plan for cbtsv1

## Executive Summary

This plan aligns **cbtsv1** (solver lab) to the **Defined Coherence** canon in `coherence-framework`. The guiding principle: **coherence-framework owns the definition; cbtsv1 consumes it.**

The canonical coherence functional is:
```
𝔠(x) = ⟨r̃(x), W r̃(x)⟩,  where r̃ = S⁻¹r
```

## Current State Analysis

### What's Already Good
- ✅ Solver residual machinery (Hamiltonian/Momentum constraints) is solid
- ✅ Ledger framework exists with receipt chaining
- ✅ Numerical stack (numpy/numba) is intact
- ✅ Structure is clean enough to patch

### What's Misaligned
- ❌ `gr_coherence.py` owns solver-local coherence logic (damping, projection)
- ❌ Constraint norms computed internally, not as canonical residual blocks
- ❌ No explicit `ResidualBlock` structure with scale/weight
- ❌ Receipts log `eps_H`, `eps_M`, `debt_decomposition` but not canonical `coherence_value`
- ❌ No hash + summary per block in receipts

---

## Phase 1: Add Coherence Framework Dependency

### 1.1 Create Vendor Stub for Canonical Coherence Core

Create: `src/cbtsv1/vendor/coherence_framework/`

```
src/cbtsv1/vendor/coherence_framework/__init__.py
src/cbtsv1/vendor/coherence_framework/coherence/
src/cbtsv1/vendor/coherence_framework/coherence/__init__.py
src/cbtsv1/vendor/coherence_framework/coherence/core.py
```

**`core.py`** must define:
```python
# Canonical ResidualBlock per Defined Coherence
@dataclass
class ResidualBlock:
    name: str                    # e.g., "hamiltonian", "momentum"
    vector: np.ndarray            # Raw residual vector
    scale: float                 # Block-specific scale factor
    weight: float                # Block-specific weight (diagonal covariance entry)
    
    # Computed properties (computed at construction)
    dim: int                     # Vector dimension
    l2: float                   # L2 norm
    linf: float                 # L-infinity norm
    hash: str                   # SHA256 of canonical vector bytes

@dataclass  
class CoherenceResult:
    coherence_value: float        # 𝔠 = ⟨r̃, W r̃⟩
    blocks: dict[str, ResidualBlock]
    covariance_model: str       # e.g., "diag"

def compute_coherence(blocks: dict[str, ResidualBlock]) -> CoherenceResult:
    """
    Canonical coherence computation.
    
    For each block:
        r̃_i = scale_i * vector_i  (apply scale)
        contribution_i = weight_i * ||r̃_i||²  (apply weight, L2 squared)
    
    Total: 𝔠 = Σ_i contribution_i
    """
    ...
```

### 1.2 Update Requirements

Add to `requirements.txt`:
```
# When coherence-framework is published, replace vendor with:
# coherence-framework>=1.0.0
```

---

## Phase 2: Create Residual Block Adapter for GR

Create: `src/cbtsv1/solvers/gr/defined_coherence_blocks.py`

```python
"""
GR Residual Block Adapter

Converts cbtsv1 GR state into canonical ResidualBlocks for coherence computation.
This is the ONLY place where cbtsv1 computes residual blocks.
"""

from cbtsv1.vendor.coherence_framework.coherence.core import ResidualBlock

def build_residual_blocks(fields, constraints, config) -> dict[str, ResidualBlock]:
    """
    Build canonical residual blocks from GR constraints.
    
    Args:
        fields: GRCoreFields instance
        constraints: GRConstraints instance (must have H and M computed)
        config: dict with 'blocks' containing scale/weight for each block
    
    Returns:
        dict[str, ResidualBlock] with keys: "hamiltonian", "momentum"
    """
    blocks = {}
    
    # Hamiltonian block
    hamiltonian_cfg = config.get("blocks", {}).get("hamiltonian", {"scale": 1.0, "weight": 1.0})
    hamiltonian_vector = constraints.H.flatten()
    blocks["hamiltonian"] = ResidualBlock(
        name="hamiltonian",
        vector=hamiltonian_vector,
        scale=hamiltonian_cfg["scale"],
        weight=hamiltonian_cfg["weight"]
    )
    
    # Momentum block (trace-adjusted)
    momentum_cfg = config.get("blocks", {}).get("momentum", {"scale": 1.0, "weight": 1.0})
    # M has shape (Nx, Ny, Nz, 3), flatten to (Nx*Ny*Nz*3,)
    momentum_vector = constraints.M.flatten()
    blocks["momentum"] = ResidualBlock(
        name="momentum",
        vector=momentum_vector,
        scale=momentum_cfg["scale"],
        weight=momentum_cfg["weight"]
    )
    
    return blocks


def summarize_blocks(blocks: dict[str, ResidualBlock]) -> dict:
    """
    Create audit-friendly summary of blocks (no large arrays).
    
    Returns:
        dict with per-block: dim, l2, linf, hash
    """
    summary = {}
    for name, block in blocks.items():
        summary[name] = {
            "dim": block.dim,
            "l2": block.l2,
            "linf": block.linf,
            "hash": block.hash,
            "scale": block.scale,
            "weight": block.weight
        }
    return summary
```

---

## Phase 3: Replace Internal Coherence Computation

### 3.1 Create Coherence Integration Module

Create: `src/cbtsv1/solvers/gr/coherence_integration.py`

```python
"""
Coherence Integration Layer

Replaces solver-local coherence with canonical compute_coherence().
"""

from cbtsv1.vendor.coherence_framework.coherence.core import compute_coherence
from .defined_coherence_blocks import build_residual_blocks, summarize_blocks

def compute_gr_coherence(fields, constraints, config) -> dict:
    """
    Compute canonical coherence for GR state.
    
    Args:
        fields: GRCoreFields
        constraints: GRConstraints (with H, M computed)
        config: coherence config dict
    
    Returns:
        dict with:
            - coherence_value: float
            - blocks: dict summary (dim, l2, linf, hash)
            - raw_result: CoherenceResult (for debugging)
    """
    blocks = build_residual_blocks(fields, constraints, config)
    result = compute_coherence(blocks)
    
    return {
        "coherence_value": result.coherence_value,
        "blocks": summarize_blocks(blocks),
        "raw_result": result
    }
```

### 3.2 Update gr_stepper.py

In `GRStepper.step_ufe()` method, replace constraint norm logging with:

```python
# OLD (remove):
# ledgers['eps_H'] = float(self.constraints.eps_H)
# ledgers['eps_M'] = float(self.constraints.eps_M)

# NEW:
from cbtsv1.solvers.gr.coherence_integration import compute_gr_coherence

# After constraint residuals are computed
coherence_result = compute_gr_coherence(
    self.fields, 
    self.constraints, 
    self.coherence_config
)

ledgers['coherence_value'] = coherence_result['coherence_value']
ledgers['residual_blocks'] = coherence_result['blocks']
```

---

## Phase 4: Update Ledger/Receipts Schema

### 4.1 Update gr_ledger.py

Modify `emit_step_receipt()` to include:

```python
receipt = {
    ...
    'coherence_value': coherence_value,  # NEW
    'residual_blocks': residual_blocks,  # NEW (dict with dim, l2, linf, hash)
    ...
}
```

### 4.2 Create Receipt Schema

Create: `schemas/coherence_receipt.schema.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Coherence Receipt",
  "type": "object",
  "required": ["step", "coherence_value", "residual_blocks"],
  "properties": {
    "step": {"type": "integer"},
    "coherence_value": {"type": "number"},
    "residual_blocks": {
      "type": "object",
      "properties": {
        "hamiltonian": {
          "type": "object",
          "required": ["dim", "l2", "linf", "hash", "scale", "weight"],
          "properties": {
            "dim": {"type": "integer"},
            "l2": {"type": "number"},
            "linf": {"type": "number"},
            "hash": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
            "scale": {"type": "number"},
            "weight": {"type": "number"}
          }
        },
        "momentum": {
          "type": "object",
          "required": ["dim", "l2", "linf", "hash", "scale", "weight"],
          "properties": {
            "dim": {"type": "integer"},
            "l2": {"type": "number"},
            "linf": {"type": "number"},
            "hash": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
            "scale": {"type": "number"},
            "weight": {"type": "number"}
          }
        }
      }
    }
  }
}
```

---

## Phase 5: Create Coherence Config File

Create: `config/defined_coherence_gr.json`

```json
{
  "version": "1.0.0",
  "covariance_model": "diag",
  "blocks": {
    "hamiltonian": {
      "scale": 1.0,
      "weight": 1.0,
      "description": "Hamiltonian constraint H = R + K² - K_ij K^ij - 2Λ"
    },
    "momentum": {
      "scale": 1.0, 
      "weight": 1.0,
      "description": "Momentum constraint M^i = D_j(K^ij - γ^ij K)"
    }
  },
  "tests": {
    "minkowski_tolerance": 1e-10,
    "schwarzschild_tolerance": 1e-6
  }
}
```

---

## Phase 6: Add Alignment Tests

### 6.1 Core Coherence Test (Smoke)

Create: `tests/test_coherence_alignment.py`

```python
"""
Test canonical coherence computation matches expected values.
"""

import numpy as np
from cbtsv1.vendor.coherence_framework.coherence.core import (
    ResidualBlock, compute_coherence
)

def test_compute_coherence_identity():
    """Zero residual should give zero coherence."""
    block = ResidualBlock(
        name="test",
        vector=np.zeros(100),
        scale=1.0,
        weight=1.0
    )
    result = compute_coherence({"test": block})
    assert result.coherence_value == 0.0

def test_compute_coherence_scaled():
    """Scaled residual should scale coherence quadratically."""
    vec = np.array([1.0, 2.0, 3.0])
    block = ResidualBlock(
        name="test",
        vector=vec,
        scale=2.0,
        weight=1.0
    )
    result = compute_coherence({"test": block})
    # r̃ = 2 * vec = [2, 4, 6]
    # ||r̃||² = 4 + 16 + 36 = 56
    # weight * ||r̃||² = 56
    assert np.isclose(result.coherence_value, 56.0)
```

### 6.2 Minkowski GR Test

```python
def test_minkowski_coherence_zero():
    """Minkowski state should have near-zero coherence."""
    from cbtsv1.solvers.gr.gr_solver import GRSolver
    from cbtsv1.solvers.gr.coherence_integration import compute_gr_coherence
    import json
    
    # Load config
    with open("config/defined_coherence_gr.json") as f:
        config = json.load(f)
    
    solver = GRSolver(16, 16, 16)
    solver.init_minkowski()
    
    # Compute coherence
    result = compute_gr_coherence(
        solver.fields, 
        solver.constraints, 
        config
    )
    
    # Should be near zero for Minkowski
    assert result['coherence_value'] < 1e-10, \
        f"Minkowski coherence {result['coherence_value']} > 1e-10"
```

### 6.3 Artificial Residual Test

```python
def test_artificial_residual_coherence():
    """Inject known residual, verify computed coherence matches hand calculation."""
    from cbtsv1.solvers.gr.coherence_integration import compute_gr_coherence
    
    # Create mock fields/constraints with known residual
    mock_fields = Mock()
    mock_constraints = Mock()
    mock_constraints.H = np.ones((8, 8, 8)) * 0.1  # 0.1 everywhere
    mock_constraints.M = np.zeros((8, 8, 8, 3))
    
    config = {
        "blocks": {
            "hamiltonian": {"scale": 1.0, "weight": 1.0},
            "momentum": {"scale": 1.0, "weight": 1.0}
        }
    }
    
    result = compute_gr_coherence(mock_fields, mock_constraints, config)
    
    # Expected: ||0.1||² * 1.0 * 1.0 = 0.01 * N (where N = 8*8*8)
    expected = 0.01 * 512
    assert np.isclose(result['coherence_value'], expected, rtol=1e-10)
```

---

## Phase 7: Update Documentation

Create: `docs/framework/defined_coherence_alignment.md`

```markdown
# Defined Coherence Alignment

cbtsv1 aligns to the **Defined Coherence** canon from `coherence-framework`.

## Canonical Definition

The coherence functional is:

𝔠(x) = ⟨r̃(x), W r̃(x)⟩,  where r̃ = S⁻¹r

Where:
- **r**: Vector of constraint residuals
- **S**: Scaling matrix (per-block)
- **W**: Weight matrix (diagonal covariance)
- **r̃**: Scaled residual vector

## cbtsv1 Residual Blocks

For GR evolution, cbtsv1 produces two residual blocks:

| Block | Physics | Vector Shape |
|-------|---------|--------------|
| `hamiltonian` | H = R + K² - K_ij K^ij - 2Λ | (Nx, Ny, Nz) |
| `momentum` | M^i = D_j(K^ij - γ^ij K) | (Nx, Ny, Nz, 3) |

## Scale/Weight Configuration

See `config/defined_coherence_gr.json` for declared scales and weights.

## Integration Point

Canonical coherence is computed in `gr_stepper.py` after constraint residuals are computed:

```python
from cbtsv1.solvers.gr.coherence_integration import compute_gr_coherence

coherence_result = compute_gr_coherence(fields, constraints, config)
ledgers['coherence_value'] = coherence_result['coherence_value']
```

## Reference

- Canonical definition: `coherence-framework/docs/coherence/01_canonical_definition.md`
```

---

## Phase 8: Deprecate Old Coherence Code

### 8.1 Mark for Deprecation

**gr_coherence.py** - Add deprecation warning:
```python
import warnings
warnings.warn(
    "gr_coherence.py is deprecated. Use coherence_integration.py for canonical coherence.",
    DeprecationWarning,
    stacklevel=2
)
```

**gr_loc.py** - The LoC operator serves a different purpose (constraint damping). Keep it but document that it does NOT compute canonical coherence.

### 8.2 Remove Solver-Owned Coherence Math

After testing confirms the new path works:
- Remove any remaining direct computation of "constraint norm" as a scalar
- Ensure all coherence values flow through `compute_coherence()`

---

## Implementation Order

1. ✅ Analysis complete
2. 📝 Create vendor stub (`core.py`)
3. 📝 Create `defined_coherence_blocks.py` adapter
4. 📝 Create `coherence_integration.py`
5. 📝 Update `gr_stepper.py` to use new integration
6. 📝 Update `gr_ledger.py` receipt schema
7. 📝 Create `config/defined_coherence_gr.json`
8. 📝 Add alignment tests
9. 📝 Create documentation
10. 📝 Deprecate old code

---

## Success Criteria

After this patch, cbtsv1:

- ✅ Produces explicit residual blocks (hamiltonian, momentum)
- ✅ Declares scale + weight per block in config
- ✅ Calls canonical `compute_coherence()`
- ✅ Logs canonical `coherence_value` into ledger
- ✅ Stores block summaries + hash in receipts
- ✅ Passes Minkowski coherence ≈ 0 test
- ✅ Passes artificial residual coherence test
