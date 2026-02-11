# Logic Alignment Review: cbtsv1 Implementation vs. Coherence Framework Theorems

## Executive Summary

Analysis of 15 modified cbtsv1 source files against formal theorems and axioms from coherence-framework parent repo reveals **3 critical alignment gaps** and **7 implementation recommendations**.

**Status:** ⚠️ **Partial alignment** - Core gate/receipt mechanisms exist but lack formal verification structure.

---

## Axiom Compliance Matrix

| Axiom | Requirement | cbtsv1 Implementation | Gap | Severity |
|-------|-------------|---------------------|-----|----------|
| **A1: Observability** | Coherence from current state only, no future peeking | `gr_gates.py` checks current residuals | ✓ | ✓ Pass |
| **A2: Attribution** | Changes decomposable into named causes | `gr_receipts.py` tracks events but not cause decomposition | ❌ | Critical |
| **A3: Bounded correction** | All corrective actions have declared bounds | Bounds hardcoded in `gr_gates.py` (eps_H_warn=5e-4, etc.), not externalized | ⚠️ | High |
| **A4: Affordability decision** | Returns accept/retry/abort with reasons | `check_gates()` returns `(accepted, hard_fail, reasons)` but structure unclear | ⚠️ | High |
| **A5: Reproducibility** | Same inputs/config/seed yield equivalent receipts | Receipts hashed but tolerance thresholds not exported | ⚠️ | Medium |

---

## File-by-File Alignment Analysis

### Core Gate Logic (`src/core/gr_gates.py`)

**Issues:**
1. **Hardcoded Thresholds (Axiom A3 violation)**
   - Lines 83-90: `eps_H_warn=5e-4`, `eps_M_hard_max=1e-1` are local constants
   - Spec requires these as externalized "declared bounds" per gate policy
   - **Fix:** Move to configurable `rails_policy` dict or policy file

2. **Hysteresis State Machine (Implementation detail)**
   - Lines 103-155: Manual state tracking for `eps_H` (normal → warn → fail)
   - Not derived from canonical stability theorems
   - **Alignment:** Check against `LyapunovProperties.lean` in coherence-framework

3. **Penalty Accumulation (Axiom A2 violation)**
   - Line 111: `penalty += (eps_H - enter_warn) / enter_warn` lacks named cause attribution
   - Spec requires decomposition: conservation_defect, reconstruction_error, tool_mismatch, thrash
   - **Reference:** [`Coherence_Spec_v1_0/docs/04_Coherence_Functionals.md`](Coherence_Spec_v1_0/docs/04_Coherence_Functionals.md)

### Receipt Emission (`src/core/gr_receipts.py`)

**Issues:**
1. **Missing Debt Decomposition (Axiom A2)**
   - Receipts track `rhs_norms`, `grid`, `clocks` but not structured debt decomposition
   - Canonical form: `𝔉𝔠 = w_cons·ε² + w_rec·r_rec² + w_tool·r_tool² + w_thrash·p_thrash`
   - **Fix:** Add `debt_decomposition` field to receipts

2. **Ledger Structure (Axiom A4)**
   - Line 50, 88: `'ledgers': {}` - ledgers are empty dicts
   - Spec requires gates and decisions per ledger entry
   - **Reference:** [`Coherence_Spec_v1_0/docs/09_Receipts_Ledgers_and_Certificates.md`](Coherence_Spec_v1_0/docs/09_Receipts_Ledgers_and_Certificates.md)

### Stability Implementation (`src/nsc/stability.py`)

**Issues:**
1. **Contractive Repair Assumption (Theorem Lemma 3)**
   - Code must enforce: `𝔉𝔠(x^(k+1)) ≤ γ·𝔉𝔠(x^(k)) + b` where γ∈(0,1)
   - No explicit validation of contraction coefficient γ
   - **Fix:** Add theorem-validation check post-step

2. **Retry Cap Enforcement (Theorem Lemma 2)**
   - No explicit counter for `N_retry` limit
   - Must prevent unbounded retry loops
   - **Reference:** [`coherence_math_spine/06_stability_theorems.md`](coherence_math_spine/06_stability_theorems.md) Line 32

### Clock/Scheduler Logic (`src/core/gr_clocks.py`, `src/core/gr_scheduler.py`)

**Issues:**
1. **No Hard Invariant Validation (Theorem Lemma 1)**
   - Lines check state but don't explicitly verify `I_hard(x_k) = true` after repair
   - **Fix:** Add pre-acceptance hard invariant check

---

## Stability Theorem Verification Checklist

### Theorem: Coherence Persistence Under Gated Evolution
**Reference:** [`coherence_math_spine/06_stability_theorems.md`](coherence_math_spine/06_stability_theorems.md) Line 59

**Requirements:**
- [ ] (S1) Bounded retries: explicit `N_retry` cap enforced
- [ ] (S2) Hard legality: rails validated before use
- [ ] (S3) Contractive repair: γ coefficient validated post-step
- [ ] (Lemma 1) Hard invariants persist on accepted steps
- [ ] (Lemma 2) Bounded work: work per step ≤ (1 + N_retry) × baseline
- [ ] (Lemma 3) Debt boundedness: 𝔉𝔠 ≤ max(𝔉𝔠₀, 𝔉𝔠_max, b/(1-γ))
- [ ] (Lemma 4) Small-gain: coupled residuals satisfy bounds

**Current Status:** ❌ 0/7 explicitly validated

---

## Modified Files Impact Assessment

| File | Issue Type | Impact |
|------|-----------|--------|
| `src/core/gr_gates.py` | Threshold hardcoding, penalty attribution | High - violates A2, A3 |
| `src/core/gr_receipts.py` | Missing debt decomposition, ledger structure | Critical - violates A2, A4 |
| `src/nsc/stability.py` | No contraction validation, no retry cap | High - violates stability theorems |
| `src/core/gr_clocks.py` | No hard invariant check post-repair | Medium - violates Lemma 1 |
| `src/core/gr_scheduler.py` | No work-bound tracking | Medium - violates Lemma 2 |
| `src/core/gr_rhs.py` | Norm tracking but not canonical decomposition | Low - ancillary |
| `src/core/gr_constraints.py` | Boundary checking but not legality proof | Medium - violates S2 |
| `src/aeonic/aeonic_memory_contract.py` | Contract enforcement structure unclear | Low - needs review |
| `src/aml/aml.py` | AML logic not aligned with gate decisions | Medium - needs review |
| `src/nllc/*.py` | Compiler logic not gate-aware | Low - upstream of gates |
| `src/phaseloom/*.py` | Rail invocation not formally validated | Medium - violates S2 |

---

## Recommended Alignment Actions

### Tier 1: Critical (Must Fix)

**1. Externalize Gate Thresholds (Axiom A3)**
```python
# Current: hardcoded in gr_gates.py
eps_H_warn = 5e-4

# Target: from policy
class GatePolicy:
    thresholds = {
        'eps_H': {'warn': 5e-4, 'fail': 1e-3},
        'eps_M': {'soft': 1e-2, 'hard': 1e-1},
        # ... all thresholds
    }
```

**2. Add Named Cause Decomposition (Axiom A2)**
```python
# Target in receipts:
receipt['debt_decomposition'] = {
    'conservation_defect': eps_H**2,
    'reconstruction_error': eps_M**2,
    'tool_mismatch': eps_proj**2,
    'thrash_penalty': p_thrash
}
```

**3. Validate Contractive Repair (Theorem Lemma 3)**
```python
def validate_contraction(debt_before, debt_after, gamma=0.8, b=1e-4):
    """Verify: debt_after ≤ γ·debt_before + b"""
    assert debt_after <= gamma * debt_before + b, \
        f"Contraction failed: {debt_after} > {gamma * debt_before + b}"
```

### Tier 2: High (Should Fix)

**4. Explicit Retry Cap Enforcement (Theorem Lemma 2)**
```python
class StepExecutor:
    MAX_RETRIES = 3  # From rails_policy
    
    def step(self, ...):
        attempts = 0
        while attempts <= self.MAX_RETRIES:
            # ... attempt step
            attempts += 1
        assert attempts <= self.MAX_RETRIES
```

**5. Hard Invariant Validation (Theorem Lemma 1)**
```python
def check_hard_invariants(fields):
    """Returns True if I_hard(x) = true"""
    return (
        is_finite(fields.alpha) and
        is_positive(fields.alpha) and
        is_valid_conformal_metric(fields.gamma_sym6) and
        is_determinant_valid(fields.K_sym6)
    )
```

**6. Reproducibility Export (Axiom A5)**
```python
# In config: explicit declaration
reproducibility_config = {
    'tolerance_thresholds': {...all gate thresholds...},
    'canonical_seed': 42,
    'solver_settings': {...},
    'receipt_format_version': '1.0'
}
```

### Tier 3: Medium (Should Consider)

**7. Ledger Structure Alignment**
- Map `src/nsc/ledger_*.py` to [`Coherence_Spec_v1_0/schemas/`](Coherence_Spec_v1_0/schemas/)
- Validate against `coherence_receipt.schema.json`

**8. Rail Legality Proof (S2)**
- Document which rails in `src/phaseloom/` preserve hard invariants
- Reference [`coherence_math_spine/lean/coherence-theorems/CoherenceTheorems/GateCorrectness.lean`](coherence_math_spine/lean/coherence-theorems/CoherenceTheorems/GateCorrectness.lean)

---

## Lean Proof References

| Theorem | Location | Applies To |
|---------|----------|-----------|
| Gate Correctness | `coherence_math_spine/lean/.../GateCorrectness.lean` | `gr_gates.py` logic |
| Debt Coercivity | `coherence_math_spine/lean/.../DebtCoercivity.lean` | Penalty accumulation |
| Lyapunov Properties | `coherence_math_spine/lean/.../LyapunovProperties.lean` | Stability bounds |
| Rail Boundedness | `coherence_math_spine/lean/.../RailBoundedness.lean` | Correction actions |
| Receipt Chain | `coherence_math_spine/lean/.../ReceiptChain.lean` | Hash chain validation |

---

## Next Steps

1. **Create alignment verification suite** - test each axiom against implementation
2. **Externalize configuration** - move hardcoded bounds to policy files
3. **Add theorem validators** - post-step checks for contraction, bounds, invariants
4. **Update receipt schema** - implement named cause decomposition
5. **Document rail legality** - which rails preserve which invariants

---

## Summary Table

| Category | Pass | Fail | Partial |
|----------|------|------|---------|
| Axioms (A1-A5) | 1 | 4 | 0 |
| Stability Theorems | 0 | 7 | 0 |
| Configuration | ❌ | ❌ | ⚠️ |
| **Overall Alignment** | **10%** | - | - |
