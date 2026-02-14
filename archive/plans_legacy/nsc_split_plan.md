# NSC Split into High-Level and Low-Level Hadamard Glyph System: Implementation Plan

## Spine Overview
The Noetica Symbolic Compiler (NSC) will be split into two tiers:
- **High-Level NSC**: Retains the current glyph system for symbolic PDE definitions, policy parsing, and user-friendly authoring.
- **Low-Level Hadamard Glyph System**: A new optimized bytecode layer focused on execution efficiency, with compact opcodes, JIT compatibility, and minimal runtime overhead.

This split enables faster execution while preserving the expressive power of the glyph language. The backbone plan is structured in phases with dependencies, targeting a 2-5x speed improvement in the Minkowski Stability Test while maintaining sub-1e-12 constraint accuracy.

## Technical Specifications

### High-Level NSC (Current Glyph System)
- **Features**: Full glyph set (φ, ↻, ⊕, etc.), symbolic PDE assembly, GR policy parsing, module hashing.
- **Output**: AST, flattened glyphs, standard bytecode.
- **Purpose**: Symbolic definition of PDEs, policies, and constraints.

### Low-Level Hadamard Glyph System
- **New Concept**: "Hadamard Glyph" is a binary-encoded glyph system where each symbolic glyph is compiled to a fixed 4-byte opcode block (1 byte opcode + 3 bytes args/metadata), enabling ultra-fast dispatch and JIT compilation. Named after a hypothetical efficiency-focused compiler design.
- **Key Features**:
  - **Compact Mapping**: High-level glyphs (e.g., ∇, ∂) map to integer opcodes (0x01 for ∂, 0x02 for ∇), with fused operations (e.g., ∇² as single opcode 0x03).
  - **Register-Based VM**: Shift from stack-based to register hints for reduced overhead.
  - **JIT-Friendly**: Opcodes designed for easy translation to LLVM IR or Numba JIT.
- **Bytecode Format**:
  - Instruction: 4 bytes [opcode:1 | arg1:1 | arg2:1 | meta:1] (e.g., ∂ field -> 0x01 [field_idx] [deriv_dir] [scale]).
  - Sequences: Packed arrays for SIMD ops.
  - Trace: Compressed spans for minimal memory.
- **VM Enhancements**: Multi-tier JIT (interpreter → compiled), caching of compiled blocks, parallel dispatch for field ops.
- **Optimizations**: Dead code elimination, constant propagation, inlining of small PDE terms.

### Compilation Pipeline
High-Level NSC → Intermediate Representation (IR) → Low-Level Hadamard Bytecode → Optimized Execution.

### Integration Points
- **Runtime Interfaces**: Hadamard VM integrates with `nsc_runtime_min.py`, `gr_solver` components.
- **Solver Bindings**: Direct mapping to HPC kernels (RicciOp, LieDerivOp).
- **Constraint Checks**: Inline accuracy monitoring for stability tests.

## Phases and Dependencies

### Phase 1: Analysis and Design (Dependency: None)
- **Milestone**: Complete design documents and specs.
- **Tasks**:
  - Analyze current NSC glyph mappings and bytecode generation.
  - Define Hadamard opcode table (compact mappings).
  - Design IR lowering from high-level to low-level.
- **Deliverables**: Design doc, opcode specs.
- **Duration**: 1 week.

### Phase 2: High-Level Refinement (Dependency: Phase 1)
- **Milestone**: Enhanced high-level compiler with IR export.
- **Tasks**:
  - Extend high-level NSC to emit IR instead of direct bytecode.
  - Preserve policy parsing and PDE assembly.
- **Deliverables**: Updated `nsc.py`, IR format spec.
- **Duration**: 2 weeks.

### Phase 3: Low-Level Hadamard Development (Dependency: Phase 1, Parallel with Phase 2)
- **Milestone**: Functional Hadamard VM and bytecode generator.
- **Tasks**:
  - Implement Hadamard opcode mappings and bytecode assembler.
  - Build optimized VM with JIT hooks.
  - Add optimizations: Constant folding, dead code elimination.
- **Deliverables**: `hadamard_vm.py`, bytecode tools.
- **Duration**: 3 weeks.

### Phase 4: Integration and Compilation Pipeline (Dependency: Phase 2 & 3)
- **Milestone**: End-to-end compilation from high-level to low-level execution.
- **Tasks**:
  - Integrate IR lowering to Hadamard bytecode.
  - Update runtime to support Hadamard ops.
  - Bind to solver intrinsics.
- **Deliverables**: Unified compiler pipeline, updated runtime.
- **Duration**: 2 weeks.

### Phase 5: Testing and Benchmarks (Dependency: Phase 4)
- **Milestone**: Validated with Minkowski Stability Test, documented improvements.
- **Tasks**:
  - Port Minkowski test to use Hadamard pipeline.
  - Run benchmarks: Execution time, constraint accuracy (eps_H, eps_M < 1e-12).
  - Compare to baseline: 2-5x speed gain, no accuracy loss.
- **Deliverables**: Benchmark reports, test suite updates.
- **Duration**: 1 week.

## Testing Benchmarks
- **Minkowski Stability Test**: 10,000+ steps in flat spacetime, monitor eps_H, eps_M.
- **Performance Metrics**: Wall time per step, memory usage, JIT overhead.
- **Accuracy Checks**: Constraint residuals vs. baseline tolerances.

## Expected Improvements
- **Speed**: 2-5x faster execution via optimized bytecode and JIT.
- **Accuracy**: Maintained at 1e-12 for constraints, improved numerical stability in long runs.
- **Efficiency**: Reduced bytecode size by 30-50%, lower VM dispatch costs.

## Risk Mitigation
- **Backwards Compatibility**: High-level NSC remains unchanged for users.
- **Fallback**: Option to compile high-level directly to old bytecode if issues arise.
- **Validation**: Extensive testing against Python baselines.

## Mermaid Diagram: Phase Dependencies
```mermaid
graph TD
    A[Phase 1: Analysis & Design] --> B[Phase 2: High-Level Refinement]
    A --> C[Phase 3: Low-Level Hadamard Dev]
    B --> D[Phase 4: Integration]
    C --> D
    D --> E[Phase 5: Testing & Benchmarks]