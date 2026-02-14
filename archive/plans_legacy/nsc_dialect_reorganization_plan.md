# NSC Dialect Reorganization Plan: Comprehensive Domain Taxonomy

**Version:** 2.1  
**Date:** 2026-02-01  
**Status:** IMPLEMENTED

---

## 1. Executive Summary

### Current State
NSC currently has flat dialect categories based on physics applications:
- `NSC_GR` (General Relativity)
- `NSC_NS` (Navier-Stokes)
- `NSC_YM` (Yang-Mills)
- `NSC_Time` (LoC-Time Protocol)

### Problem
The flat structure doesn't scale and mixes concerns:
- Physics domain with semantic models
- No clear separation between mathematical foundations and applications
- Limited extensibility for new domains

### Proposed Solution
**Hierarchical taxonomy** inspired by academic disciplines with three levels:
1. **Level 1: Discipline** (Mathematics, Physics, Computation, Linguistics)
2. **Level 2: Field** (Within each discipline)
3. **Level 3: Subfield/Application** (Specific implementations)

---

## 2. Complete NSC Taxonomy

### 2.1 Discipline Overview

```
NSC/
├── NSC_mathematics/        # Mathematical foundations
├── NSC_physics/           # Physical theories & applications
├── NSC_computation/       # Computer science & algorithms
└── NSC_linguistics/       # Language & formal systems
```

### 2.2 Detailed Field Structure

#### NSC_MATHEMATICS

```
mathematics/
├── algebra/
│   ├── linear/            # Linear algebra, matrices, vector spaces
│   ├── abstract/          # Groups, rings, fields, modules
│   ├── representation/    # Group representations, characters
│   └── homological/       # Category theory, homology, cohomology
│
├── analysis/
│   ├── real/              # Real analysis, measure theory
│   ├── complex/           # Complex analysis, conformal mapping
│   ├── functional/        # Banach/Hilbert spaces, operators
│   └── harmonic/          # Fourier analysis, wavelets
│
├── geometry/
│   ├── euclidean/         # Euclidean geometry
│   ├── differential/      # Manifolds, connections, curvature
│   ├── algebraic/         # Algebraic varieties, schemes
│   ├── riemannian/        # Riemannian metrics, geodesics
│   └── symplectic/        # Symplectic forms, Hamiltonian systems
│
├── topology/
│   ├── general/           # Point-set topology
│   ├── algebraic/         # Homotopy, homology
│   ├── differential/      # Differential topology
│   └── geometric/         # Geometric topology
│
├── dynamics/
│   ├── odes/              # Ordinary differential equations
│   ├── pdes/              # Partial differential equations
│   ├── chaos/             # Chaotic systems, attractors
│   └── ergodic/           # Ergodic theory
│
├── probability/
│   ├── stochastic/        # Stochastic processes
│   ├── statistics/        # Statistical inference
│   └── measure_theoretic/ # Measure-theoretic probability
│
├── optimization/
│   ├── convex/            # Convex optimization
│   ├── variational/       # Calculus of variations
│   └── discrete/          # Combinatorial optimization
│
└── numerical/
    ├── approximation/     # Function approximation
    ├── integration/       # Numerical integration
    ├── linear_solvers/    # Linear system solvers
    └── eigensystems/      # Eigenvalue problems
```

#### NSC_PHYSICS

```
physics/
├── classical/
│   ├── mechanics/         # Lagrangian/Hamiltonian mechanics
│   ├── electromagnetism/  # Maxwell's equations
│   ├── thermodynamics/    # Heat, entropy, statistical mechanics
│   └── optics/            # Geometric and wave optics
│
├── relativity/
│   ├── special/           # Special relativity, Lorentz transforms
│   ├── general/           # General relativity, Einstein equations
│   └── cosmology/         # Cosmological models, dark energy
│
├── quantum/
│   ├── mechanics/         # Wave mechanics, operators
│   ├── field_theory/      # QFT, Feynman diagrams
│   ├── many_body/         # Many-body physics
│   └── quantum_info/      # Quantum computing, entanglement
│
├── gauge_theory/
│   ├── yang_mills/        # Non-Abelian gauge theories
│   ├── electroweak/       # Electroweak unification
│   └── quantum_chromo/    # QCD, color charge
│
├── fluid_dynamics/
│   ├── incompressible/    # Navier-Stokes (incompressible)
│   ├── compressible/      # Compressible flows
│   ├── turbulence/        # Turbulence models
│   └── plasma/            # Plasma physics, MHD
│
├── condensed_matter/
│   ├── solid_state/       # Crystal lattices, phonons
│   ├── superconductivity/ # BCS theory, Cooper pairs
│   └── topological/       # Topological insulators
│
└── particle/
    ├── standard_model/    # Particle physics
    ├── beyond_standard/   # BSM physics
    └── lattice/           # Lattice field theory
```

#### NSC_COMPUTATION

```
computation/
├── theory/
│   ├── automata/          # Finite automata, Turing machines
│   ├── complexity/        # P, NP, complexity classes
│   └── computability/     # Decidability, Gödel theorems
│
├── algorithms/
│   ├── sorting/           # Sorting algorithms
│   ├── search/            # Search algorithms
│   ├── graph/             # Graph algorithms
│   └── optimization/      # Optimization algorithms
│
├── languages/
│   ├── compilers/         # Lexing, parsing, code generation
│   ├── type_theory/       # Type systems, dependent types
│   └── semantics/         # Operational/denotational semantics
│
├── systems/
│   ├── architecture/      # CPU, memory, instruction sets
│   ├── operating_systems/ # Processes, scheduling, memory management
│   ├── distributed/       # Consensus, CAP theorem, RPC
│   └── networks/          # Protocols, routing, security
│
├── graphics/
│   ├── rendering/         # Ray tracing, rasterization
│   ├── geometry/          # Mesh processing, subdivision
│   └── animation/         # Keyframes, physics-based animation
│
├── ai_ml/
│   ├── machine_learning/  # Supervised/unsupervised learning
│   ├── deep_learning/     # Neural networks, CNNs, RNNs
│   ├── reinforcement/     # RL, policy gradient
│   └── nlp/               # Natural language processing
│
└── verification/
    ├── formal_methods/    # Model checking, theorem proving
    ├── testing/           # Unit testing, fuzzing
    └── static_analysis/   # Data flow, abstract interpretation
```

#### NSC_LINGUISTICS

```
linguistics/
├── theoretical/
│   ├── phonology/         # Sound systems
│   ├── morphology/        # Word formation
│   ├── syntax/            # Sentence structure
│   └── semantics/         # Meaning
│
├── computational/
│   ├── nlp/               # Tokenization, tagging, parsing
│   ├── generation/        # Text generation
│   ├── understanding/     # Question answering, summarization
│   └── speech/            # Speech recognition/synthesis
│
├── logic/
│   ├── propositional/     # Propositional logic
│   ├── predicate/         # Predicate logic
│   ├── modal/             # Modal logic
│   └── linear/            # Linear logic
│
└├── formal_systems/
    ├── proof_theory/      # Natural deduction, sequent calculus
    ├── model_theory/      # Structures and models
    └── set_theory/        # ZFC, ordinals, cardinals
```

---

## 3. NSC-Specific Organization

### 3.1 Domain Categories (Practical Subset)

For NSC's current scope, we'll implement a practical subset:

```
NSC/
├── NSC_geometry/           # Geometry & differential structure (subset of mathematics)
│   ├── gr/                 # General Relativity
│   ├── riemann/            # Riemannian geometry
│   ├── symplectic/         # Symplectic geometry (for Hamiltonian systems)
│   └── ym/                 # Yang-Mills (gauge geometry)
│
├── NSC_fluids/             # Fluid dynamics (subset of physics)
│   ├── navier_stokes/      # Incompressible NS
│   ├── euler/              # Ideal fluid (ν=0)
│   └── mhd/                # Magnetohydrodynamics
│
├── NSC_algebra/            # Algebraic structures
│   ├── linear/             # Linear algebra operations
│   ├── lie/                # Lie algebras
│   ├── tensor/             # Tensor operations
│   └── clifford/           # Geometric algebra / clifford algebra
│
├── NSC_analysis/           # Analysis operations
│   ├── calculus/           # Differential/integral calculus
│   ├── pdes/               # PDE operators
│   ├── spectral/           # Fourier, eigenvalue analysis
│   └── variational/        # Calculus of variations
│
├── NSC_numerical/          # Numerical methods
│   ├── stencils/           # Finite difference stencils
│   ├── quadrature/         # Integration rules
│   ├── solvers/            # Linear/nonlinear solvers
│   └── stability/          # Stability analysis
│
├── NSC_quantum/            # Quantum mechanics/field theory
│   ├── quantum_mech/       # Wave functions, operators
│   ├── quantum_field/      # Field operators
│   └── lattice_gauge/      # Lattice gauge theory
│
└── NSC_compile/            # Compilation & execution (computation)
    ├── lexer/              # Lexical analysis
    ├── parser/             # Parsing
    ├── type_checker/       # Type checking
    ├── nir/                # Intermediate representation
    ├── vm/                 # Virtual machine
    └── ledger/             # Receipts, invariants, gates
```

### 3.2 Compilation Target Layer (Semantic Models)

```
NSC_compile/
├── calc/                   # Calculus semantics
├── alg/                    # Algebra semantics
├── geo/                    # Geometry semantics
├── disc/                   # Discrete semantics
├── ledger/                 # Audit/invariant semantics
└── exec/                   # Runtime semantics
```

---

## 4. Module Structure

### 4.1 Proposed Directory Layout

```
src/nsc/
├── __init__.py
│
├── core/                   # Core compiler infrastructure
│   ├── __init__.py
│   ├── lexer.py
│   ├── parser.py
│   ├── ast.py
│   ├── type_checker.py
│   ├── types.py
│   ├── lower_pir.py
│   └── flatten.py
│
├── domains/                # Mathematical/physical domains
│   ├── __init__.py
│   │
│   ├── geometry/           # Geometry domain
│   │   ├── __init__.py
│   │   ├── gr/
│   │   │   ├── __init__.py
│   │   │   ├── types.py        # Metric, ExtrinsicK, BSSN
│   │   │   ├── operators.py    # Christoffel, Ricci
│   │   │   ├── invariants.py   # Hamiltonian/momentum constraints
│   │   │   └── lowering.py     # To NIR
│   │   ├── riemann/
│   │   │   ├── __init__.py
│   │   │   ├── types.py
│   │   │   └── operators.py
│   │   ├── symplectic/
│   │   │   └── ...
│   │   └── ym/
│   │       ├── __init__.py
│   │       ├── types.py        # Connection, FieldStrength
│   │       └── operators.py
│   │
│   ├── fluids/             # Fluids domain
│   │   ├── __init__.py
│   │   ├── navier_stokes/
│   │   │   ├── __init__.py
│   │   │   ├── types.py        # Velocity, Pressure
│   │   │   ├── operators.py    # Advection, diffusion
│   │   │   └── invariants.py   # Div-free, energy
│   │   ├── euler/
│   │   │   └── ...
│   │   └── mhd/
│   │       └── ...
│   │
│   ├── algebra/            # Algebra domain
│   │   ├── __init__.py
│   │   ├── linear/
│   │   │   ├── __init__.py
│   │   │   ├── types.py
│   │   │   └── operators.py
│   │   ├── lie/
│   │   │   ├── __init__.py
│   │   │   ├── types.py
│   │   │   └── operators.py    # Commutator, adjoint
│   │   ├── tensor/
│   │   │   ├── __init__.py
│   │   │   ├── types.py
│   │   │   └── operators.py    # Trace, det, symmetrize
│   │   └── clifford/
│   │       └── ...
│   │
│   ├── analysis/           # Analysis domain
│   │   ├── __init__.py
│   │   ├── calculus/
│   │   │   └── operators.py    # Grad, div, curl, laplacian
│   │   ├── pdes/
│   │   │   └── operators.py    # PDE residual operators
│   │   ├── spectral/
│   │   │   └── operators.py    # FFT, eigenvalues
│   │   └── variational/
│   │       └── operators.py    # Functional derivatives
│   │
│   ├── numerical/          # Numerical methods domain
│   │   ├── __init__.py
│   │   ├── stencils/
│   │   │   ├── __init__.py
│   │   │   ├── types.py
│   │   │   └── operators.py
│   │   ├── quadrature/
│   │   │   └── ...
│   │   ├── solvers/
│   │   │   └── ...
│   │   └── stability/
│   │       └── ...
│   │
│   └── quantum/            # Quantum domain
│       ├── __init__.py
│       ├── quantum_mech/
│       ├── quantum_field/
│       └── lattice_gauge/
│
├── models/                 # Semantic model targets
│   ├── __init__.py
│   ├── calc/               # Calculus model
│   │   ├── __init__.py
│   │   └── operators.py
│   ├── alg/                # Algebra model
│   │   ├── __init__.py
│   │   └── operators.py
│   ├── geo/                # Geometry model
│   │   ├── __init__.py
│   │   └── operators.py
│   ├── disc/               # Discrete model
│   │   ├── __init__.py
│   │   └── operators.py
│   ├── ledger/             # Ledger model (audit)
│   │   ├── __init__.py
│   │   ├── gates.py
│   │   ├── receipts.py
│   │   ├── invariants.py
│   │   └── validator.py
│   └── exec/               # Execution model (runtime)
│       ├── __init__.py
│       ├── vm.py
│       ├── opcodes.py
│       ├── scheduler.py
│       └── jit.py
│
└── nir/                    # Intermediate representation
    ├── __init__.py
    ├── assemble_pde.py
    └── flatten.py
```

---

## 5. Dialect Declaration Format

### 5.1 New Format (Domain + Models)

```nsc
# Single domain with multiple models
@domain("NSC_geometry.gr")
@model("GEO")
@model("CALC")
@model("LEDGER")
@model("EXEC")

# Or compound declaration
@compile(
    domain=NSC_geometry.gr,
    models=[GEO, CALC, LEDGER, EXEC],
    invariants=[hamiltonian_constraint, momentum_constraint]
)

# Multi-domain declaration
@domain("NSC_algebra.lie")
@domain("NSC_geometry.gr")
```

### 5.2 Backward Compatibility

```python
# compat.py - Backward compatibility aliases
NSC_GR = NSC_geometry.gr
NSC_NS = NSC_fluids.navier_stokes
NSC_YM = NSC_geometry.ym
NSC_Time = NSC_models.ledger.exec
```

---

## 6. Implementation Phases

### Phase 1: Infrastructure (Week 1)
- Create `domains/` and `models/` directories
- Move core files to `core/`
- Create __init__.py files

### Phase 2: Domain Implementation (Weeks 2-4)
- Implement `NSC_geometry.gr`
- Implement `NSC_fluids.navier_stokes`
- Implement `NSC_algebra.lie`
- Implement `NSC_numerical` modules

### Phase 3: Model Implementation (Weeks 5-6)
- Implement `models/ledger`
- Implement `models/exec`
- Implement `models/calc`, `models/geo`

### Phase 4: Deprecation (Week 7)
- Add deprecation warnings
- Create compatibility layer
- Update documentation

---

## 7. Breaking Changes Summary

| Old Path | New Path |
|----------|----------|
| `nsc.alg_types` | `nsc.domains.algebra.tensor.types` |
| `nsc.ledger_gates` | `nsc.models.ledger.gates` |
| `nsc.stencils` | `nsc.domains.numerical.stencils` |
| `nsc.exec_vm` | `nsc.models.exec.vm` |

---

## 8. Benefits

1. **Scalable**: New domains easily added under discipline hierarchy
2. **Clear**: Separation of mathematical domains from compilation targets
3. **Comprehensive**: Covers all major areas of mathematics, physics, computation
4. **Extensible**: Users can add custom domains
5. **Maintainable**: Focused modules with clear boundaries

---

## References

- Original dialect specs: [`plans/nsc_gr_dialect_spec.md`](plans/nsc_gr_dialect_spec.md)
- Core spec: [`specifications/nsc_m3l_v1.md`](specifications/nsc_m3l_v1.md)

---

## 9. Implementation Summary (v2.1)

### Implemented Structure

```
src/nsc/
├── __init__.py                    # Main package with unified domain exports
│
├── types.py                       # Core types: Scalar, Vector, Tensor, Field
│
├── domains/                       # Mathematical/physical domains
│   ├── __init__.py                # Exports: NSC_geometry, NSC_fluids, NSC_algebra,
│   │                               #           NSC_analysis, NSC_numerical, NSC_quantum
│   │
│   ├── geometry/
│   │   ├── __init__.py            # Exports: NSC_geometry, NSC_GR, NSC_riemann, NSC_YM
│   │   ├── geometry.py            # Unified NSC_geometry dialect with types
│   │   ├── gr/                    # General Relativity subdomain
│   │   ├── riemann/               # Riemannian geometry subdomain
│   │   └── ym/                    # Yang-Mills gauge theory subdomain
│   │
│   ├── fluids/
│   │   ├── __init__.py            # Exports: NSC_fluids, NSC_NS
│   │   ├── fluids.py              # Unified NSC_fluids dialect with types
│   │   └── navier_stokes.py       # Navier-Stokes subdomain
│   │
│   ├── algebra/
│   │   ├── __init__.py            # Exports: NSC_algebra, NSC_lie
│   │   ├── algebra.py             # Unified NSC_algebra dialect
│   │   └── lie.py                 # Lie algebra subdomain
│   │
│   ├── analysis/
│   │   ├── __init__.py            # Exports: NSC_analysis
│   │   └── analysis.py            # Analysis domain (calculus operators)
│   │
│   ├── numerical/
│   │   ├── __init__.py            # Exports: NSC_numerical, NSC_stencils
│   │   ├── numerical.py           # Unified NSC_numerical dialect
│   │   └── stencils.py            # Finite difference stencils
│   │
│   └── quantum/
│       ├── __init__.py            # Exports: NSC_quantum
│       └── quantum.py             # Quantum domain (wave functions, operators)
│
└── models/                        # Semantic model targets
    ├── __init__.py                # Exports: NSC_ledger, NSC_VM
    │
    ├── ledger/
    │   └── ledger.py              # NSC_Ledger_Dialect with gates, receipts
    │
    └── exec/
        ├── __init__.py
        └── vm.py                  # NSC_VM with physics kernels (GRAD, DIV, CURL, etc.)
```

### Key Features

1. **Unified Dialects**: Each domain has a top-level dialect (e.g., `NSC_geometry`)
   that exports types, operators, and invariants for that domain.

2. **Subdomain Support**: Subdomains are accessible via the main domain exports
   (e.g., `NSC_geometry.gr` for GR-specific functionality).

3. **VM Physics Kernels**: Implemented in `src/nsc/models/exec/vm.py`:
   - `GRAD`: Gradient computation using numpy.gradient
   - `DIV_OP`: Divergence computation
   - `CURL`: Curl computation (∇×v)
   - `LAPLACIAN`: Laplacian computation
   - `CHRISTOFFEL`: Christoffel symbols from metric
   - `RICCI`: Ricci tensor computation

### Usage Example

```python
import nsc

# Import unified domains
from nsc.domains import NSC_geometry, NSC_fluids, NSC_algebra

# Access types and operators
metric = NSC_geometry.type_hierarchy['Metric']()
christoffel = NSC_geometry.operators['christoffel']

# VM execution with physics kernels
from nsc.models.exec.vm import NSC_VM, BytecodeProgram, OpCode

program = BytecodeProgram(name='gradient_test')
program.add_instruction(Instruction(OpCode.PUSH, immediate=np.array([1.0, 2.0, 3.0])))
program.add_instruction(Instruction(OpCode.GRAD))
program.add_instruction(Instruction(OpCode.HALT))

vm = NSC_VM(program)
vm.run()
```
