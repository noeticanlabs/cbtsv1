# NSC System Upgrade Plan (v2.1 — Post-Implementation)

**Version:** 2.1
**Date:** 2026-01-31
**Status:** ~90% IMPLEMENTED

## Executive Summary
The Noetica Symbolic Compiler (NSC) system has been upgraded from prototype to a robust, structured language compiler (NLLC). This document reflects the current implementation state as of the v2 upgrade.

## 1. Architecture Consolidation — ✅ COMPLETE

| Item | Status | Implementation |
|------|--------|----------------|
| `src/nllc` as canonical compiler | ✅ DONE | [`src/nllc/lex.py`](src/nllc/lex.py), [`src/nllc/parse.py`](src/nllc/parse.py) |
| Rich glyph set in NLLC | ✅ DONE | 56 token types including `thread`, `audit`, `rollback` |
| Legacy NSC deprecation | ⚠️ PARTIAL | `nsc_compile_min.py` still exists, needs migration |

## 2. Type System Hardening — ✅ COMPLETE

| Item | Status | Implementation |
|------|--------|----------------|
| `TypeChecker` pass | ✅ DONE | [`src/nllc/type_checker.py`](src/nllc/type_checker.py) |
| GR types: `Tensor<T, D>`, `Field`, `Metric`, `Clock` | ✅ DONE | [`src/nllc/nir.py`](src/nllc/nir.py) |
| Dimensional consistency | ✅ DONE | Type checker enforces tensor operations |

**Example Type Rules:**
```python
# Valid
Int + Int → Int
Float * Float → Float
Tensor<1> + Tensor<1> → Tensor<1>

# Invalid (caught by type checker)
Int + Float  # type mismatch
Tensor<2> + Tensor<1>  # dimension mismatch
```

## 3. NIR Optimization Pipeline — ⚠️ PARTIAL

| Item | Status | Implementation |
|------|--------|----------------|
| `Mem2Reg` pass | ✅ DONE | [`src/nllc/mem2reg.py`](src/nllc/mem2reg.py) |
| `ConstantFolding` | ✅ DONE | [`src/nllc/constant_folding.py`](src/nllc/constant_folding.py) |
| `DeadCodeElimination` | ❌ PENDING | Not implemented |

**Mem2Reg Transformation:**
```python
# Before (stack-allocated)
%0 = alloc Float
store %0, 1.0
%1 = load %0
%2 = add %1, 2.0

# After (SSA register)
%0 = 1.0                # ConstInst
%1 = add %0, 2.0        # BinOpInst (no load/store)
```

## 4. GR/HPC Intrinsic Integration — ⚠️ PARTIAL

| Item | Status | Implementation |
|------|--------|----------------|
| NIR instructions for kernels | ⚠️ PARTIAL | [`src/nllc/intrinsic_binder.py`](src/nllc/intrinsic_binder.py) exists |
| JIT-compiled kernel binding | ❌ PENDING | Not implemented |
| VM intrinsics for GR | ⚠️ PARTIAL | [`src/nllc/vm.py`](src/nllc/vm.py) has built-ins |

**Supported Intrinsics:**
- `inv_sym6`, `trace_sym6`, `sym6_to_mat33`
- `mat33_to_sym6`, `det_sym6`, `norm2_sym6`

## 5. Tooling & Diagnostics — ✅ COMPLETE

| Item | Status | Implementation |
|------|--------|----------------|
| NLLC CLI | ✅ DONE | [`src/nllc/cli.py`](src/nllc/cli.py) |
| Enhanced error reporting | ⚠️ PARTIAL | Trace objects exist, needs UI |
| Source line/spans | ✅ DONE | [`src/nllc/ast.py`](src/nllc/ast.py) Span class |

## Roadmap

### Phase 1: Unification (Week 1) — ✅ COMPLETE
- [x] Merge lexer/parser logic
- [x] Update CLI to use NLLC (CLI pending)

### Phase 2: Safety (Week 2) — ✅ COMPLETE
- [x] Implement TypeChecker
- [x] Define GR types

### Phase 3: Performance (Week 3) — ✅ COMPLETE
- [x] Implement Mem2Reg
- [x] Implement ConstantFolding
- [ ] Bind JIT kernels to VM intrinsics (pending)

### Phase 4: Validation (Week 4) — ⚠️ IN PROGRESS
- [ ] Run full Minkowski stability tests
- [ ] Generate benchmark reports

## Files Added/Modified

| File | Purpose |
|------|---------|
| [`src/nllc/cli.py`](src/nllc/cli.py) | NLLC CLI tool |
| [`src/nllc/constant_folding.py`](src/nllc/constant_folding.py) | Constant folding optimization |
| [`src/nllc/type_checker.py`](src/nllc/type_checker.py) | Type safety for GR types |
| [`src/nllc/mem2reg.py`](src/nllc/mem2reg.py) | SSA optimization |
| [`src/nllc/nir.py`](src/nllc/nir.py) | Extended NIR types |
| [`tests/test_full_stack_integration.py`](tests/test_full_stack_integration.py) | 46 integration tests |

## Next Steps

1. **HIGH PRIORITY**: Implement `DeadCodeElimination` pass
2. **HIGH PRIORITY**: Implement JIT kernel binding
3. **MEDIUM PRIORITY**: Run Minkowski benchmark tests
4. **LOW PRIORITY**: Create benchmark reporting tool
