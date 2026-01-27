# NSC Compilation Pipeline Design: High-Level NSC to Hadamard Bytecode

## Overview
This design outlines the compilation pipeline for splitting the Noetica Symbolic Compiler (NSC) into high-level symbolic PDE definition and low-level optimized numerical execution via the Hadamard bytecode system. The pipeline ensures symbolic PDE definitions are preserved at the high level while enabling efficient numerical optimizations at the low level, particularly for General Relativity (GR) simulations such as Minkowski stability tests.

## Pipeline Stages

### 1. High-Level NSC Parsing and Glyph Flattening
- **Input**: `.nsc` files (e.g., `minkowski_rhs.nsc`) containing glyph-based PDE definitions.
- **Process**:
  - Lexical analysis using `src/nsc/lex.py` to tokenize the glyph language.
  - Parsing into AST using `src/nsc/parse.py` to build a structured representation (Program, Sentence, Phrase, Atom, Group).
  - Glyph flattening via `src/nsc/flatten.py` to extract FlatGlyph objects with symbols like `φ`, `∂`, `∇`, preserving paths and spans for traceability.
- **Output**: Flattened glyph list with symbolic PDE structures (e.g., field names, operators, parameters).
- **Purpose**: Retain expressive, symbolic authoring for PDEs, policies, and GR constraints. Symbolic definitions allow for policy parsing and module hashing without numerical evaluation.

### 2. Symbolic PDE Assembly
- **Input**: Flattened glyphs.
- **Process**:
  - Assemble PDEs using `src/nsc/assemble_pde.py` to construct symbolic expressions (e.g., RHS bundles for GR evolution).
  - Preserve symbolic forms: e.g., `rhs_gamma_sym6 = compute_gr_rhs(fields, lambda_val, sources_enabled)` remains symbolic until IR export.
- **Output**: Assembled symbolic PDE structures.
- **Purpose**: Maintain high-level symbolic integrity, enabling user-friendly authoring and policy enforcement.

### 3. Intermediate IR Export (.nscir.json)
- **Input**: Assembled symbolic PDEs.
- **Process**:
  - Serialize to JSON IR format (e.g., `minkowski_rhs.nscir.json`) with fields like `module`, `op`, `in`, `out`, `params`, `ir_hash`.
  - Structure includes compartment effects (`read/write` on `S_PHY`), dimension, boundary conditions, and expression kinds (e.g., `gr_rhs_bundle`).
- **Output**: `.nscir.json` file preserving symbolic definitions (e.g., field structs, param defaults).
- **Integration**: Uses existing IR format; no changes needed for compatibility.

### 4. IR Lowering to PIR (Program Intermediate Representation)
- **Input**: `.nscir.json`.
- **Process**:
  - Lower to PIR using `src/nsc/lower_pir.py` and `src/solver/pir.py`.
  - Translate symbolic expressions into operator-based PIR (e.g., PIRProgram with fields and operators like diffusion, source, curvature_coupling).
  - Map GR-specific ops (e.g., Ricci tensor, Lie derivatives) to PIR operators.
- **Output**: PIRProgram object with structured operators for numerical computation.
- **Purpose**: Bridge symbolic IR to executable representation, preparing for bytecode compilation.

### 5. PIR to Hadamard Bytecode Compilation
- **Input**: PIRProgram.
- **Process**:
  - Use `src/hadamard/compiler.py` (HadamardCompiler) to translate PIR operators to Hadamard bytecode.
  - Map PIR operators to opcodes (e.g., `diffusion` → `∇²`, `source` → `⊕`).
  - Assign field indices via field_map for compact arg references.
  - Assemble 4-byte instructions: [opcode:1B | arg1:1B | arg2:1B | meta:1B] (e.g., `∂` on field 0, dir 0, order 1: `0x01 0x00 0x00 0x01`).
- **Output**: Hadamard bytecode (bytes).
- **Integration**: Leverages new HadamardAssembler for packing instructions.

### 6. Low-Level Hadamard Execution with Optimizations
- **Input**: Bytecode.
- **Process**:
  - Execute on `src/hadamard/vm.py` (HadamardVM) with numpy field arrays.
  - Dispatch opcodes to optimized numerical ops (e.g., partial_deriv_op using finite differences or spectral methods for GR).
  - Apply low-level optimizations: JIT compilation (via numba), hotspot detection, constant folding, dead code elimination.
  - Numerical execution for GR: Efficient handling of tensor ops, SIMD-friendly packed sequences, parallel dispatch for field computations.
- **Output**: Computed RHS bundles or constraint checks.
- **Purpose**: Optimize for speed (2-5x improvement) and accuracy (sub-1e-12 eps_H, eps_M in Minkowski tests).

## Key Design Aspects

### Preservation of Symbolic PDE Definitions
- High-level stages (1-3) keep PDEs symbolic: glyphs, expressions, and structures remain abstract, enabling policy parsing, hashing, and user authoring without premature numerical instantiation.
- IR export serializes symbols intact, allowing analysis and transformation before lowering.

### Numerical Execution Optimizations
- Low-level stages (5-6) focus on efficiency: 4-byte compact bytecode reduces dispatch overhead; register-hinted VM (stack-based with field indices) minimizes memory access; JIT for hotspots (e.g., repeated derivative ops in GR evolutions) targets HPC performance.
- For GR simulations: Optimized intrinsics for RicciOp, LieDerivOp; caching of compiled blocks; parallel execution across grid points.

### Integration Points
- Existing IR: Direct use of `.nscir.json` as intermediate stage, integrating with current tools (e.g., `test_minkowski.py`, `gr_gate_policy.nscir.json`). The IR format is consumed by `src/nsc/lower_pir.py` for PIR generation, ensuring seamless transition without modifying existing IR exporters.
- Hadamard Assembler: New component (`src/hadamard/assembler.py`) for bytecode generation, using opcode mappings from `HADAMARD_OPCODES`. Extensible via addition to the dict (e.g., GR-specific like `ricci: 0x20`, `lie: 0x21`, `constraint: 0x22`, `gauge: 0x23`). Assembler packs instructions into bytes, compatible with HadamardVM's unpack.
- Runtime: HadamardVM (`src/hadamard/vm.py`) augments `nsc_runtime_min.py` by providing a bytecode-based execution layer. Integrates with `gr_solver` via field arrays (e.g., maps to `gr_core_fields.py`), and binds to HPC kernels (e.g., RicciOp from `gr_solver/elliptic/solver.py`). VM state includes jit_cache for compiled hotspots, linking to `phaseloom_gr_adapter.py` for GR simulations.
- Compilation Flow: High-level NSC compilation now optionally targets IR export instead of direct bytecode. New compiler script orchestrates IR → PIR → Hadamard bytecode. Existing bytecode paths remain for fallback.
- Testing Integration: Update `test_minkowski.py` to use pipeline, comparing outputs with `test_python_baseline.py`. Benchmarks via `timing_script.py` for wall time, memory in `gr_solver/spectral/cache.py`.

### Enabling Efficient GR Simulations
- Minkowski Tests: Pipeline compiles symbolic RHS definitions to optimized bytecode, enabling long-run stability checks (10,000+ steps) with maintained accuracy and reduced wall time.
- Scalability: Design supports HPC via JIT and parallel dispatch, aligning with `plans/hpc_integration_plan.md`.

## Pipeline Diagram
```mermaid
graph TD
    A[High-Level NSC (.nsc)] --> B[Parse & Flatten Glyphs]
    B --> C[Assemble Symbolic PDEs]
    C --> D[Export Intermediate IR (.nscir.json)]
    D --> E[Lower to PIR]
    E --> F[Compile to Hadamard Bytecode]
    F --> G[Execute on Hadamard VM]
    G --> H[Optimized Numerical Results]
```

## Risk Mitigation and Validation
- Backwards Compatibility: High-level unchanged; fallback to old bytecode if needed.
- Validation: Port Minkowski test to pipeline, benchmark vs. baseline (Python pure).
- Expected Gains: 2-5x speed, <1e-12 accuracy, 30-50% bytecode reduction.

## Conclusion
This pipeline splits NSC effectively, preserving symbolic power at high-level while optimizing numerical execution at low-level via Hadamard. It enables efficient GR simulations, starting with Minkowski tests.