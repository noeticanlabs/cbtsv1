l# GR Solver API Reference

This document provides detailed reference for all public classes and methods in the GR solver module (`gr_solver`). The API is designed for relativistic PDE evolution using the Universal Field Equation (UFE) framework with PhaseLoom control and Aeonic memory.

## Table of Contents

- [GRSolver](#grsolver)
- [GRStepper](#grstepper)
- [GRCoreFields](#grcorefields)
- [GRGeometry](#grgeometry)
- [GRConstraints](#grconstraints)
- [GRGauge](#grgauge)
- [GRPhaseLoomOrchestrator](#grphaseloomorchestrator)
- [Utility Functions](#utility-functions)
- [Examples](#examples)

## GRSolver

Main solver class for GR evolution. Manages initialization, fields, geometry, constraints, and orchestrates evolution with PhaseLoom control.

### `__init__(Nx, Ny, Nz, dx=1.0, dy=1.0, dz=1.0, c=1.0, Lambda=0.0, log_level=logging.INFO, log_file=None)`

Initializes the GR solver with grid parameters and physical constants.

**Parameters:**
- `Nx, Ny, Nz` (int): Grid dimensions
- `dx, dy, dz` (float): Spatial grid spacings
- `c` (float): Speed of light (for relativistic units)
- `Lambda` (float): Cosmological constant
- `log_level` (int): Logging level
- `log_file` (str, optional): Log file path

**Returns:** None

### `init_minkowski()`

Initializes fields to Minkowski spacetime with small perturbations for testing.

**Parameters:** None  
**Returns:** None

### `run(T_max, dt_max=None)`

Executes evolution loop until T_max or violation.

**Parameters:**
- `T_max` (float): Maximum evolution time
- `dt_max` (float, optional): Maximum time step

**Returns:** None (logs progress and stops on violations)

## GRStepper

Handles time stepping for GR evolution using RK4 with UFE formulation.

### `__init__(fields, geometry, constraints, gauge, memory_contract=None, phaseloom=None, aeonic_mode=True)`

**Parameters:**
- `fields` (GRCoreFields): Field container
- `geometry` (GRGeometry): Geometry computations
- `constraints` (GRConstraints): Constraint monitoring
- `gauge` (GRGauge): Gauge evolution
- `memory_contract` (AeonicMemoryContract, optional): Memory for auditing
- `phaseloom` (PhaseLoom27, optional): Control system
- `aeonic_mode` (bool): Enable preallocation for performance

**Returns:** None

### `step_ufe(dt, t=0.0)`

Performs RK4 step for UFE evolution.

**Parameters:**
- `dt` (float): Time step size
- `t` (float): Current time

**Returns:** None (updates fields in-place)

### `compute_rhs(t=0.0, slow_update=True)`

Computes right-hand sides for ADM/BSSN evolution.

**Parameters:**
- `t` (float): Current time
- `slow_update` (bool): Update slow fields (phi, Z, Z_i)

**Returns:** None (stores RHS in instance variables)

### `apply_damping()`

Applies constraint damping via exponential decay of K_sym6.

**Parameters:** None  
**Returns:** None

## GRCoreFields

Container for all GR field variables using efficient sym6 storage for symmetric tensors.

### `__init__(Nx, Ny, Nz, dx=1.0, dy=1.0, dz=1.0)`

**Parameters:** Grid dimensions and spacings as above.  
**Returns:** None

### `init_minkowski()`

Sets fields to Minkowski values.  
**Parameters:** None  
**Returns:** None

### `bssn_decompose()`

Decomposes ADM fields into BSSN variables.  
**Parameters:** None  
**Returns:** None

### `bssn_recompose()`

Recombines BSSN into ADM fields.  
**Parameters:** None  
**Returns:** None

## GRGeometry

Computes geometric quantities: Christoffels, Ricci tensor, scalar curvature.

### `__init__(fields)`

**Parameters:**
- `fields` (GRCoreFields): Field container

**Returns:** None

### `compute_christoffels()`

Computes Christoffel symbols Γ^k_ij and Gamma^i.  
**Parameters:** None  
**Returns:** None (stores in self.christoffels, self.Gamma)

### `compute_ricci()`

Computes Ricci tensor R_ij using BSSN conformal method.  
**Parameters:** None  
**Returns:** None (stores in self.ricci)

### `compute_scalar_curvature()`

Computes scalar curvature R = γ^ij R_ij.  
**Parameters:** None  
**Returns:** None (stores in self.R)

### `lie_derivative_gamma(gamma_sym6, beta)`

Computes Lie derivative L_β γ_ij.  
**Parameters:**
- `gamma_sym6` (np.ndarray): Metric in sym6 form
- `beta` (np.ndarray): Shift vector

**Returns:** sym6 array of Lie derivative

### `lie_derivative_K(K_sym6, beta)`

Computes Lie derivative L_β K_ij.  
**Parameters:** Similar to gamma.  
**Returns:** sym6 array

## GRConstraints

Monitors Hamiltonian and momentum constraints.

### `__init__(fields, geometry)`

**Parameters:**
- `fields` (GRCoreFields)
- `geometry` (GRGeometry)

**Returns:** None

### `compute_hamiltonian()`

Computes H = R + K^2 - K_ij K^ij - 2Λ.  
**Parameters:** None  
**Returns:** None (stores eps_H)

### `compute_momentum()`

Computes M_i = D^j K_ji - D_i K.  
**Parameters:** None  
**Returns:** None (stores eps_M)

### `compute_residuals()`

Computes both H and M residuals.  
**Parameters:** None  
**Returns:** None

## GRGauge

Handles gauge evolution (lapse and shift).

### `__init__(fields, geometry)`

Similar initialization.

### `evolve_lapse(dt)`

Evolves α using 1+log slicing.  
**Parameters:** `dt` (float)  
**Returns:** None

### `evolve_shift(dt)`

Evolves β^i using Gamma-driver.  
**Parameters:** `dt` (float)  
**Returns:** None

## GRPhaseLoomOrchestrator

Orchestrates evolution with PhaseLoom control, rails/gates enforcement, and rollback.

### `__init__(fields, geometry, constraints, gauge, stepper, ledger, memory_contract=None, phaseloom=None, eps_H_target=1e-10, eps_M_target=1e-10, m_det_min=0.2, aeonic_mode=True)`

**Parameters:**
- Similar to stepper, plus:
- `ledger` (GRLedger): Logging system
- `eps_H_target, eps_M_target` (float): Target residuals
- `m_det_min` (float): Minimum metric determinant

**Returns:** None

### `run_step(dt_max)`

Executes one controlled step with PhaseLoom arbitration.  
**Parameters:** `dt_max` (float, optional)  
**Returns:** tuple (dt_used, dominant_thread, rail_violation)

## Utility Functions

Located in `gr_core_fields.py`:

### `sym6_to_mat33(sym6)`

Convert sym6 tensor to 3x3 matrix.  
**Parameters:** `sym6` (np.ndarray, shape ...,6)  
**Returns:** np.ndarray (shape ...,3,3)

### `mat33_to_sym6(mat)`

Convert 3x3 matrix to sym6.  
**Parameters:** `mat` (np.ndarray, shape ...,3,3)  
**Returns:** np.ndarray (shape ...,6)

### `inv_sym6(sym6, det_floor=1e-14)`

Inverse of sym6 tensor with determinant check.  
**Parameters:** `sym6` (np.ndarray)  
**Returns:** sym6 inverse

### `trace_sym6(sym6, inv_sym6)`

Compute γ^ij A_ij.  
**Parameters:** Both sym6 arrays  
**Returns:** scalar field

### `repair_spd_eigen_clamp(sym6, lambda_floor=1e-8)`

Repair SPD matrix by clamping eigenvalues.  
**Parameters:** `sym6` input  
**Returns:** tuple (repaired_sym6, lambda_min_pre, lambda_min_post)

## Examples

### Basic Minkowski Evolution

```python
from gr_solver import GRSolver

# Initialize solver
solver = GRSolver(Nx=32, Ny=32, Nz=32, dx=0.1, dy=0.1, dz=0.1)
solver.init_minkowski()

# Run evolution
solver.run(T_max=1.0, dt_max=0.01)
```

### Custom Initial Data

```python
import numpy as np
from gr_solver.gr_core_fields import GRCoreFields, SYM6_IDX

fields = GRCoreFields(32, 32, 32, 0.1, 0.1, 0.1)
# Set custom metric
fields.gamma_sym6[..., SYM6_IDX['xx']] = 1 + perturbation
# ... other fields

geometry = GRGeometry(fields)
geometry.compute_christoffels()
```

See also: [Project Lexicon](project_lexicon_canon_v1_2.md) for term definitions, [PhaseLoom Control](phaseloom_control.md) for advanced usage.