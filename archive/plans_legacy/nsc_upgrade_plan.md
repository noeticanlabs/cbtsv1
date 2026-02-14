# NSC System Upgrade Plan

## Executive Summary
The Noetica Symbolic Compiler (NSC) system is transitioning from a prototype phase to a robust, structured language compiler (NLLC). The current state involves legacy logic that was previously spread across multiple prototype implementations, now consolidated into `src/nllc` (AST, NIR, VM). This upgrade plan targets the unification of these systems, introducing static typing, optimization passes, and deep integration with the GR/HPC solver stack.

## 1. Architecture Consolidation
**Goal**: Establish `src/nllc` as the canonical compiler pipeline.
- **Action**: Port the rich glyph set (e.g., `ℋ`, `𝓜`, `𝔊`) and policy parsing logic to `src/nllc/lex.py` and `src/nllc/parse.py`.
- **Action**: Deprecate `nsc_compile_min.py` and legacy NSC logic, replacing them with NLLC-based equivalents.
- **Benefit**: Removes code duplication and provides a single source of truth for language semantics.

## 2. Type System Hardening
**Goal**: Ensure safety and correctness before runtime.
- **Action**: Implement a `TypeChecker` pass that runs on the AST before NIR lowering.
- **Action**: Introduce specific types for GR entities: `Tensor<T, D>`, `Field`, `Metric`, `Scalar`.
- **Action**: Enforce dimensional consistency in tensor operations at compile time.
- **Benefit**: Catches physics errors (e.g., adding a scalar to a tensor) early.

## 3. NIR Optimization Pipeline
**Goal**: Improve runtime performance of the VM.
- **Action**: Implement `Mem2Reg` pass to convert stack-allocated variables (introduced by the recent control-flow fix) back to SSA registers where possible.
- **Action**: Add `ConstantFolding` and `DeadCodeElimination` passes.
- **Benefit**: Reduces instruction count and memory overhead in the VM.

## 4. GR/HPC Intrinsic Integration
**Goal**: Seamlessly bridge NLLC scripts with the high-performance GR solver.
- **Action**: Define NIR instructions for heavy-lifting kernels (e.g., `RicciOp`, `LieDerivOp`, `EvolveGaugeOp`).
- **Action**: Map these instructions directly to JIT-compiled functions in the VM, bypassing the interpreter for inner loops.
- **Benefit**: Allows NLLC to orchestrate HPC kernels without interpreter overhead.

## 5. Tooling & Diagnostics
**Goal**: Improve developer experience.
- **Action**: Update CLI to drive the `src/nllc` pipeline.
- **Action**: Enhance VM error reporting using `Trace` objects to point to exact source lines/spans.
- **Benefit**: Easier debugging and usage.

## Roadmap

### Phase 1: Unification (Week 1)
- Merge lexer/parser logic.
- Update CLI to use NLLC.

### Phase 2: Safety (Week 2)
- Implement TypeChecker.
- Define GR types.

### Phase 3: Performance (Week 3)
- Implement Mem2Reg and basic optimizations.
- Bind JIT kernels to VM intrinsics.

### Phase 4: Validation (Week 4)
- Run full Minkowski stability tests using the upgraded NLLC pipeline.
