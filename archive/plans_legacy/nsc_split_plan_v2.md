# NSC Split into High-Level and Low-Level Hadamard Glyph System (v2 — Post-Implementation)

**Version:** 2.0
**Date:** 2026-01-27
**Status:** ~85% IMPLEMENTED

## Spine Overview
The Noetica Symbolic Compiler (NSC) has been split into two tiers:
- **High-Level NSC**: Retains the glyph system for symbolic PDE definitions, policy parsing, and user-friendly authoring.
- **Low-Level Hadamard Glyph System**: Optimized bytecode layer with compact opcodes (4-byte format), JIT compatibility, and minimal runtime overhead.

This split enables faster execution while preserving the expressive power of the glyph language.

## Technical Specifications

### High-Level NSC (Current Glyph System)
| Feature | Status | Implementation |
|---------|--------|----------------|
| Full glyph set (φ, ↻, ⊕, etc.) | ✅ DONE | [`src/nsc/lex.py`](src/nsc/lex.py) |
| Symbolic PDE assembly | ✅ DONE | [`src/nsc/assemble_pde.py`](src/nsc/assemble_pde.py) |
| GR policy parsing | ✅ DONE | [`src/nsc/parse.py`](src/nsc/parse.py) |
| Module hashing | ✅ DONE | NIR module structure |
| Output: AST, flattened glyphs | ✅ DONE | [`src/nsc/flatten.py`](src/nsc/flatten.py) |

### Low-Level Hadamard Glyph System
| Feature | Status | Implementation |
|---------|--------|----------------|
| 4-byte bytecode format | ✅ DONE | [`src/hadamard/assembler.py`](src/hadamard/assembler.py) |
| Compact opcode mapping | ✅ DONE | 0x01=∂, 0x02=∇, 0x03=∇², etc. |
| Register-based VM | ✅ DONE | [`src/hadamard/vm.py`](src/hadamard/vm.py) |
| JIT-friendly opcodes | ⚠️ PARTIAL | Opcodes defined, JIT not implemented |
| Multi-tier JIT | ❌ PENDING | Not implemented |
| Dead code elimination | ❌ PENDING | Not implemented |
| Constant propagation | ❌ PENDING | Not implemented |

**Bytecode Format:**
```
Instruction: 4 bytes [opcode:1 | arg1:1 | arg2:1 | meta:1]
Example: ∂ field -> 0x01 [field_idx] [deriv_dir] [scale]
```

**Hadamard Opcode Table:**
| Opcode | Glyph | Meaning |
|--------|-------|---------|
| 0x01 | ∂ | Partial derivative |
| 0x02 | ∇ | Gradient |
| 0x03 | ∇² | Laplacian |
| 0x04 | φ | Scalar field |
| 0x05 | ↻ | Curvature |
| 0x06 | ⊕ | Injection |
| 0x07 | ⊖ | Dissipation |
| 0x20 | ricci | Ricci tensor |
| 0x21 | lie | Lie derivative |

## Compilation Pipeline

```
.nsc file → NSC Parser → Flatten → Assemble PDE → PIR
                                                      ↓
                                            NSCToHadamardCompiler
                                                      ↓
                                            Hadamard Bytecode (4-byte)
                                                      ↓
                                            Hadamard VM Execution
```

**Implementation:** [`src/nsc/nsc_to_hadamard.py`](src/nsc/nsc_to_hadamard.py)

## Phases and Dependencies

### Phase 1: Analysis and Design — ✅ COMPLETE
- [x] Analyze current NSC glyph mappings
- [x] Define Hadamard opcode table
- [x] Design IR lowering

### Phase 2: High-Level Refinement — ✅ COMPLETE
- [x] Extend high-level NSC to emit IR
- [x] Preserve policy parsing and PDE assembly

### Phase 3: Low-Level Hadamard Development — ⚠️ IN PROGRESS
- [x] Implement Hadamard opcode mappings
- [x] Build bytecode assembler
- [x] Build Hadamard VM
- [ ] Add JIT hooks (pending)
- [ ] Add constant folding (pending)
- [ ] Add dead code elimination (pending)

### Phase 4: Integration and Compilation Pipeline — ✅ COMPLETE
- [x] Integrate IR lowering to Hadamard bytecode
- [x] Update runtime to support Hadamard ops
- [x] Bind to solver intrinsics

### Phase 5: Testing and Benchmarks — ⚠️ IN PROGRESS
- [x] Port tests to use Hadamard pipeline
- [ ] Run Minkowski stability tests
- [ ] Generate benchmark reports
- [ ] Document 2-5x speed improvement

## Testing Benchmarks

| Benchmark | Status | Implementation |
|-----------|--------|----------------|
| Minkowski Stability Test | ⚠️ NEEDS VERIFICATION | Tests exist, needs full run |
| Performance metrics | ⚠️ NEEDS VERIFICATION | No benchmark reports yet |
| Constraint accuracy | ✅ VERIFIED | eps_H, eps_M < 1e-12 (tested) |

## Expected Improvements (Target vs Current)

| Metric | Target | Current Status |
|--------|--------|----------------|
| Speed improvement | 2-5x | ⚠️ Not measured |
| Constraint accuracy | 1e-12 | ✅ Verified |
| Bytecode size reduction | 30-50% | ⚠️ Not measured |

## Risk Mitigation

| Risk | Mitigation | Status |
|------|------------|--------|
| Backwards compatibility | High-level NSC unchanged | ✅ OK |
| Fallback option | Compile to old bytecode | ⚠️ Partial |
| Validation | Extensive testing | ✅ 46 tests pass |

## Files Added/Modified

| File | Purpose |
|------|---------|
| [`src/nsc/nsc_to_hadamard.py`](src/nsc/nsc_to_hadamard.py) | NSC→Hadamard compiler |
| [`src/hadamard/assembler.py`](src/hadamard/assembler.py) | 4-byte bytecode assembler |
| [`src/hadamard/vm.py`](src/hadamard/vm.py) | Hadamard VM |
| [`src/hadamard/compiler.py`](src/hadamard/compiler.py) | Bytecode compiler |
| [`tests/test_full_stack_integration.py`](tests/test_full_stack_integration.py) | 46 integration tests |

## Next Steps

1. **HIGH PRIORITY**: Run Minkowski stability benchmark (10,000+ steps)
2. **HIGH PRIORITY**: Measure actual speed improvement
3. **MEDIUM PRIORITY**: Implement JIT hooks for Hadamard VM
4. **MEDIUM PRIORITY**: Implement constant folding optimization
5. **LOW PRIORITY**: Implement dead code elimination
