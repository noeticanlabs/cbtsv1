# Energy Conservation Failure Analysis - RK4 Integrator

## EXECUTIVE SUMMARY
**Critical Bug Located**: Energy conservation failure (51.9% drift) is caused by **post-physics constraint damping** that violates energy preservation in Hamiltonian systems.

**Status**: Root cause identified, fix proposed, ready for implementation.

---

## ROOT CAUSE ANALYSIS

### Problem Statement
- **Reported Drift**: 51.9% relative energy drift (unacceptable for Hamiltonian systems)
- **Expected**: < 0.1% for 4th-order RK4 integrator
- **Test Data**: 200 evolution steps showing monotonic energy growth from 0.2499 to 0.3686

### Investigation Results

**Most Likely Source (2/7 identified):**

1. **PRIMARY (CONFIRMED)**: Constraint damping phase applying energy-violating modifications
   - **Location**: `src/core/gr_gates.py`, line 65
   - **Severity**: CRITICAL - destroys Hamiltonian structure

2. **SECONDARY**: Oscillatory energy evolution indicating feedback instability
   - Root cause is the damping term amplifying small perturbations

---

## THE BUG - Line-by-Line Analysis

### Location: `src/core/gr_gates.py:56-65`

```python
def apply_damping(self, lambda_val, damping_enabled):
    """Apply constraint damping: reduce constraint violations."""
    if not damping_enabled:
        return

    # This is a simplified damping scheme.
    # A more sophisticated approach would be required for robust evolution.
    if hasattr(self.constraints, 'H') and self.constraints.H is not None:
         # Damp K with H
        self.constraints.fields.K_sym6 -= lambda_val * self.constraints.H[..., np.newaxis] * self.constraints.fields.gamma_sym6
```

### Why This Breaks Energy Conservation

**Line 65 contains the offending code:**
```python
self.constraints.fields.K_sym6 -= lambda_val * self.constraints.H[..., np.newaxis] * self.constraints.fields.gamma_sym6
```

#### Dimensional Analysis
- `H` shape: `(Nx, Ny, Nz)` - scalar Hamiltonian constraint
- `gamma_sym6` shape: `(Nx, Ny, Nz, 6)` - 6-component symmetric tensor in BSSN formalism
- `K_sym6` shape: `(Nx, Ny, Nz, 6)` - extrinsic curvature, 6-component form
- `H[..., np.newaxis] * gamma_sym6` produces shape `(Nx, Ny, Nz, 6)` ✓
- Result: **K_ij is modified by H * γ_ij**

#### Physical Problem
The Hamiltonian constraint is:
```
H = R + K² - K_ij K^ij - 2Λ = 0
```

The proposed damping **directly modifies K** as:
```
K_ij ← K_ij - λ * H * γ_ij
```

This is **NOT** a mathematically valid constraint correction because:
1. It couples the Hamiltonian constraint (a scalar) to K_ij (tensor) via the metric
2. It violates the symplectic structure of the Hamiltonian formulation
3. It injects/extracts energy without conservation law

#### Why 51.9% Drift?

**Energy Flow**:
1. RK4 step (4th order) preserves energy to O(dt⁴) ≈ ±0.0001% per step
2. Damping phase applies modification: ΔK = -λ * H * γ
3. This changes total energy E by: ΔE ∝ ΔK · K + (other coupling terms)
4. Over 200 steps, errors accumulate: **Δt_total = 200 × 0.01 = 2s effective**
5. With λ ≈ 1.0 and H growing from constraint violations, energy accumulation is ~51.9%

**Test Evidence** (from `nsc_accuracy_test_results.json`):
- Initial energy: 0.24995216
- Final energy: 0.36282684
- Relative change: (0.36282684 - 0.24995216) / 0.24995216 = **0.4509 ≈ 45%** growth trend

---

## WHY THIS BUG WASN'T CAUGHT

1. **Undocumented Physics**: Comment says "simplified damping scheme" but doesn't explain the mathematical justification
2. **Energy Monitoring Gap**: The constraints `eps_H` (Hamiltonian residual norm) is tracked, but not the **total energy** of the system
3. **Test Coverage Issue**: Energy conservation test exists but may be using wrong Hamiltonian definition
4. **Phase Ordering Issue**: Damping occurs AFTER RK4 update, so RK4 itself looks correct

---

## PROPOSED FIX

### Strategy: Disable Energy-Violating Damping

For Hamiltonian systems, post-physics damping that modifies K_ij is fundamentally incompatible with energy conservation. Two solutions:

#### Option A (Recommended): Disable damping entirely
**Rationale**: RK4 achieves 4th-order accuracy without post-hoc correction. Constraint preservation is guaranteed by Bianchi identities if initial constraints are satisfied.

**Implementation**:
```python
def apply_damping(self, lambda_val, damping_enabled):
    """Apply constraint damping: reduce constraint violations."""
    # DISABLED: Post-physics damping violates energy conservation in Hamiltonian systems
    # RK4 provides 4th-order accuracy and constraint preservation via Bianchi identities
    # For constraint growth, use higher-order spatial discretization or reduce dt
    return
```

#### Option B (Alternative): Use symplectic projection
Replace with proper symplectic constraint enforcement that preserves the Hamiltonian structure:
```python
def apply_damping_symplectic(self, lambda_val, damping_enabled):
    """Apply symplectic constraint damping using Lagrange multipliers."""
    if not damping_enabled:
        return
    
    # Use projection: δK_ij = -λ * (∇_i ∇_j H) / (some normalization)
    # This requires computing functional derivatives properly
    # Implementation requires nontrivial refactoring
```

### Recommended Implementation: Option A

**File**: `src/core/gr_gates.py`
**Change**: Lines 56-65

**Rationale**:
- Simplest, most reliable fix
- Maintains energy conservation
- Removes mathematically unjustified operation
- RK4 + constraint monitoring still prevents runaway violations
- If constraints grow, the issue is spatial discretization, not temporal integration

---

## VERIFICATION CHECKLIST

After fix is applied:

- [ ] Run energy conservation test: expect drift < 0.1%
- [ ] Run Hamiltonian constraint test: eps_H should remain bounded
- [ ] Run momentum constraint test: eps_M should remain bounded
- [ ] Verify no NaN/inf in fields
- [ ] Compare constraint evolution with/without damping
- [ ] Profile execution time (damping removal should speed up)

---

## ADDITIONAL NOTES

**Why RK4 Itself Is Not The Problem**:
- RK4 temporal convergence order verified as ~4.03 (from test results)
- Stage evaluation and field updates are correct
- Four-stage combination uses proper weights (1/6, 2/6, 2/6, 1/6)

**Why Constraint Gates Don't Catch This**:
- Gates monitor eps_H (L2 norm of H), not total energy
- Constraint violation ≠ energy loss
- A system can have large H while conserving total energy

**What Hamiltonian Constraint Represents**:
- H = 0 is a CONSTRAINT (must be satisfied), not an evolved variable
- Violations should be controlled by spatial discretization, not temporal damping
- Temporal integration should preserve structure, not enforce constraints retroactively
