# NSC Multi-Model Mathematical Linguistics (NSC-M3L) v1.0

**Document Version**: 1.0  
**Created**: 2026-01-31  
**Status**: Canonical Specification

---

## 0) Core Definition

**[DEFINITION] NSC-M3L** is a formally specified glyph language such that:

1. **Syntax** is defined by an explicit grammar (EBNF).
2. **Static semantics** (attribute grammar / type rules) determine when a glyph program is *meaningful*.
3. **Denotational semantics** map the same parsed structure into one of several **semantic models**:

   - **ALG** (algebra/linear algebra/group actions)
   - **CALC** (calculus/variational/PDE operators)
   - **GEO** (geometry: manifolds/metrics/connections/curvature)
   - **DISC** (discrete numerics: grid/FEM/lattice)
   - **LEDGER** (invariants, gates, receipts, proof obligations)
   - **EXEC** (bytecode/interpreter)

**[RULE] Multi-model** means: **one surface form → many semantics**, but the compiler must emit an **audit object** that states *exactly* which model(s) were targeted and what equivalences/constraints were enforced.

---

## 1) Linguistic vs Mathematical

| Aspect | Definition | Example |
|--------|------------|---------|
| **Linguistics** | Grammar defines legal forms (strings → parse trees) | `u + v` is valid, `u +` is not |
| **Mathematical** | Meaning is a mathematical object | `⟦u + v⟧` = addition operator in some domain |
| **Multi-model** | Same parse tree denotes different objects per model | `Δ(u)` = Laplace-Beltrami (GEO) vs stencil (DISC) |

This is "compiler theory, but the semantic domain is math."

---

## 2) NSC Surface Syntax (Minimal Core)

### 2.1 Token Types

**Identifiers**: `u, v, θ, ρ, g, Γ, F, A, E, p, φ, ψ, ...`  
**Constants**: Numbers, `π`, `e`, `i`  
**Keywords**: `::`, `@model`, `@inv`, `@gate`, `⇒`, `d/dt`, `∂`

### 2.2 Operators (Canonical)

| Operator | Meaning | Models |
|----------|---------|--------|
| `∇` | Gradient / Covariant derivative | CALC, GEO, YM |
| `div` | Divergence | CALC, GEO |
| `curl` | Curl / exterior derivative dual | CALC, GEO |
| `Δ` | Laplacian / Laplace-Beltrami | CALC, GEO |
| `d/dt` | Time derivative | CALC, EXEC |
| `∂` | Partial derivative | CALC |
| `∫` | Integral | CALC, GEO |
| `⟨ , ⟩` | Inner product / pairing | ALG, CALC |
| `∘` | Composition | ALG, CALC |
| `⊗` | Tensor product | ALG, GEO |
| `[ , ]` | Commutator (Lie algebra) | ALG, YM |

### 2.3 Control / Audit Keywords

| Token | Purpose |
|-------|---------|
| `::` | Type annotation / declaration |
| `@model(...)` | Model selection clause |
| `@inv(...)` | Invariants to enforce |
| `@gate(...)` | Acceptance gate policy |
| `⇒` | Compile / lower boundary (AST → IR) |

### 2.4 EBNF Grammar

```ebnf
Program      := { Statement } ;

Statement    := Decl | Equation | Functional | Constraint | Directive ;

Decl         := Ident "::" Type [ ":" Meta ] ";" ;

Equation     := Expr "=" Expr [ ":" Meta ] ";" ;

Functional   := "J" "(" Bindings ")" ":=" Expr [ ":" Meta ] ";" ;

Constraint   := "C" "(" Ident ")" ":=" Predicate [ ":" Meta ] ";" ;

Directive    := "@model" "(" ModelList ")" ";"
              | "@inv"   "(" InvList ")" ";"
              | "@gate"  "(" GateSpec ")" ";"
              | "⇒" TargetList ";" ;

Expr         := Term { ("+"|"-") Term } ;
Term         := Factor { ("*"|"/") Factor } ;
Factor       := Atom | Op "(" Expr ")" | "(" Expr ")" ;

Atom         := Ident | Number | Tensor | FieldAccess ;

Type         := Scalar | Vector | TensorType | FieldType | FormType | OperatorType ;

Meta         := { Key "=" Value } ;

ModelList    := Model { "," Model } ;
Model        := "ALG" | "CALC" | "GEO" | "DISC" | "LEDGER" | "EXEC" ;
```

**[RULE]** This grammar is intentionally small. Everything fancy is **library macros** that expand into these core forms.

---

## 3) Static Semantics: Attribute Grammar Layer

EBNF tells you what's *parseable*. This layer tells you what's *meaningful*.

### 3.1 Semantic Types (Shared Across Models)

```python
# Base semantic types
Scalar                           # ℝ^n element
Vector                           # Tangent space element
Tensor(k, l)                     # (k, l) tensor type
Operator(Domain → Codomain)      # Bounded linear operator
Functional                       # ℝ-valued operator
Field[T]                         # Field over spacetime with values of type T
Form[p]                          # Differential p-form
BundleConnection(gauge_group)    # Connection on principal bundle (YM)
Metric                           # Riemannian/Pseudo-Riemannian metric
Manifold(dim, signature)         # Manifold with dimension and signature
LieAlgebra(name)                 # Lie algebra for gauge groups
```

### 3.2 Required AST Node Attributes

Every AST node carries a **record** (compiler metadata):

| Attribute | Type | Description |
|-----------|------|-------------|
| `type` | SemanticType | Inferred semantic type |
| `domains_used` | Set[Model] | Subset of {ALG,CALC,GEO,DISC,LEDGER,EXEC} |
| `units` | Optional[Dimension] | Physical dimension tags |
| `regularity` | SmoothnessClass | Required smoothness (H¹, C², etc.) |
| `time_mode` | `{physical, audit, both}` | Temporal semantics |
| `invariants_required` | List[InvariantID] | Invariant IDs node depends on |
| `effects` | Set[Effect] | `{read_state, write_state, nonlocal, gauge_change}` |
| `model_tags` | Dict[Model, Tag] | Per-model semantic annotations |

### 3.3 Static Rules

**[RULE] No Hidden Regularity**  
You cannot apply `∇` or `Δ` unless the node's `regularity` supports it.

**[RULE] No Hidden Geometry**  
You cannot apply `Δ` in GEO mode unless a `Metric` is in scope.

**[RULE] No Hidden Algebra**  
You cannot use commutators `[A,B]` unless the type is Lie-algebra-valued.

### 3.4 Typing Rules (Selection)

| Expression | Condition | Result Type |
|------------|-----------|-------------|
| `∇(x)` | `x: Field[Scalar]` (CALC) | `Field[Vector]` |
| `∇(x)` | `x: Field[Vector]` (GEO, requires Metric) | `Field[Tensor(1,1)]` |
| `Δ(x)` | `x: Field[Scalar]` | `Field[Scalar]` |
| `Δ(x)` | `x: Field[Vector]` (GEO, requires connection) | `Field[Vector]` |
| `div(x)` | `x: Field[Vector]` | `Field[Scalar]` |
| `curl(x)` | `x: Field[Vector]` | `Field[Vector]` |
| `[A,B]` | `A,B: LieAlgebra(g)` | `LieAlgebra(g)` |
| `⟨x,y⟩` | `x,y: Vector` (same space) | `Scalar` |
| `∫_M f dV` | `f: Field[Scalar]`, `M: Manifold` | `Scalar` |

---

## 4) Semantic Models

### 4.1 ALG Model (Algebra / Linear Algebra)

**Semantic Domain Objects**:
- Expressions in rings/fields
- Linear maps and matrices
- Lie algebra operations (commutators)
- Symbolic simplification / normal forms

**Denotation**: `⟦Expr⟧_ALG` = algebraic expression tree with rewrite rules

**Use Cases**:
- Canonical simplification
- Invariants as algebraic identities
- Operator factoring for solvers

### 4.2 CALC Model (Calculus / Analysis / PDE / Variational)

**Semantic Domain Objects**:
- Differential operators, PDE residuals, weak forms
- Function spaces, norms, energy functionals
- Time stepping forms

**Denotation**:
- `⟦Equation⟧_CALC` → residual operator `R(u) = 0`
- `⟦Functional⟧_CALC` → `J(u)` with Euler-Lagrange extraction

**Use Cases**:
- PDE residual computation for gates
- Energy functionals for audit
- Variational derivatives

### 4.3 GEO Model (Geometry / Differential Geometry)

**Semantic Domain Objects**:
- Manifolds, charts, metrics, connections
- Curvature tensors/scalars
- Differential forms and exterior calculus

**Denotation**:
- `∇` → covariant derivative `∇^g` (Levi-Civita from metric)
- `Δ` → Laplace-Beltrami `Δ_g`
- `R` → scalar curvature or curvature form

**Use Cases**:
- GR/YM correctness (geometry is not optional)
- Coordinate-free semantics

### 4.4 DISC Model (Discrete Numerics)

**Semantic Domain Objects**:
- Grids, stencils, FEM spaces, lattice links
- Discrete operators with stability constraints

**Denotation**:
- `⟦Δ(u)⟧_DISC` → `LaplacianStencil(u, dx, scheme_id)`
- Integrals → quadrature
- Covariant derivatives → discrete connection transport

**Use Cases**:
- Finite difference / FEM compilation
- Stability analysis

### 4.5 LEDGER Model (Invariants / Gates / Receipts / Proof Obligations)

**Semantic Domain Objects**:
- Invariant IDs + acceptance conditions
- Gate policies (thresholds, hysteresis)
- Receipt schemas + hash chain logic
- Proof obligation DAG nodes

**Denotation**: Every statement yields a **Ledger Spec**:
```python
LedgerSpec = {
    "invariants_enforced": [InvariantID],
    "metrics_produced": {MetricName: Type},
    "gate_checks": {GateName: ThresholdSpec},
    "receipt_fields": [FieldSpec],
    "obligations": [ProofObligation]
}
```

### 4.6 EXEC Model (Runtime)

**Semantic Domain Objects**:
- Bytecode/IR for VM execution
- Deterministic ordering guarantees
- Stage-time schedule for LoC-Time

---

## 5) Multi-Model Compilation Pipeline

**[RULE]** Compilation = "make meaning + receipts" (not just code).

### Pipeline Stages

```
1. Parse:         source → AST
2. Normalize:     rewrite macros, canonicalize names
3. Static Check:  types + attributes + legality by model
4. Lower to PIR:  opcodes with effects and required invariants
5. Model Project: produce ALG/CALC/GEO/DISC/LEDGER/EXEC artifacts
6. Emit Receipt:  hash of (source, AST, PIR, models) + invariants
```

### Equivalence Tracking

If compiling to multiple models (e.g., GEO + DISC), must either:
- Prove equivalence (rare)
- Record comparison as **witness inequality** ("DISC approximates GEO with error ≤ ε")
- Mark as **unproven** and treat as research obligation

---

## 6) Example: Navier-Stokes in NSC-M3L

```nsc
@model(GEO, CALC, ALG, LEDGER);

M :: Manifold(3, riemannian);
g :: Metric on M;

u :: Field[Vector] on (M, t);
p :: Field[Scalar] on (M, t);
ν :: Scalar;

Eq1 := d/dt(u) + (u·∇)(u) + ∇p - ν*Δ(u) = 0;

@inv(N:INV.ns.div_free, N:INV.ns.energy_nonincreasing, 
     N:INV.clock.stage_coherence, N:INV.ledger.hash_chain_intact);
⇒ (LEDGER, CALC, GEO);
```

### Denotations

| Model | Denotation |
|-------|------------|
| **ALG** | Operator equation in noncommutative algebra (useful for factorization) |
| **CALC** | NSE residual `R(u,p)=0` + energy functional for audit |
| **GEO** | `∇` and `Δ` as covariant derivative and Laplace-Beltrami on `(M,g)` |
| **LEDGER** | Gate checks: div_free, energy_nonincreasing, stage_coherence |

---

## 7) Dialects (Overlay System)

A **dialect** is:
- Same core grammar
- Stricter static semantics
- Fixed set of invariants + required receipt fields
- Restricted operator library

### 7.1 NSC_GR (General Relativity)

```nsc
@model(GEO, CALC, LEDGER, EXEC);

M :: Manifold(3+1, lorentzian);
g :: Metric on M;
K :: Field[Tensor(0,2)] on (M, t);  # Extrinsic curvature

# Hamiltonian constraint
H := R(g) + K^2 - K_ij*K^ij = 0;

# Momentum constraint  
M_i := D_j(K^j_i - g^j_i*K) = 0;

@inv(N:INV.gr.hamiltonian_constraint, N:INV.gr.momentum_constraint,
     N:INV.gr.det_gamma_positive, N:INV.clock.stage_coherence);
⇒ (LEDGER, CALC, GEO, EXEC);
```

**Mandatory Models**: GEO, CALC, LEDGER, EXEC  
**Required**: Metric, Connection, Constraints

### 7.2 NSC_NS (Navier-Stokes)

```nsc
@model(CALC, LEDGER, EXEC);

u :: Field[Vector] on (M, t);
p :: Field[Scalar] on (M, t);
ν :: Scalar;
ρ :: Scalar;

# Incompressible NS
Eq1 := d/dt(u) + (u·∇)u - ν*Δ(u) + ∇p = 0;
Eq2 := div(u) = 0;

@inv(N:INV.ns.div_free, N:INV.ns.energy_nonincreasing,
     N:INV.ns.cfl_stability, N:INV.clock.stage_coherence);
⇒ (LEDGER, CALC, EXEC);
```

**Mandatory Models**: CALC, LEDGER, EXEC  
**GEO**: Optional but must be explicit if used

### 7.3 NSC_YM (Yang-Mills)

```nsc
@model(ALG, GEO, CALC, LEDGER, EXEC);

G :: LieAlgebra(su(N));
A :: BundleConnection(G) on (M, t);
F :: Field[Tensor(0,2; G)] on (M, t);  # Curvature 2-form

# Yang-Mills equation
D_μ(F^μν) = 0;  # In vacuum

# Gauss law constraint
D_i(E^i) = 0;  # Electric field divergence

@inv(N:INV.ym.gauss_law, N:INV.ym.bianchi_identity,
     N:INV.ym.ward_identity, N:INV.ym.gauge_condition);
⇒ (LEDGER, CALC, GEO, ALG, EXEC);
```

**Mandatory Models**: ALG, GEO, CALC, LEDGER, EXEC  
**Required**: Lie algebra, Connection, Gauge conditions

### 7.4 NSC_Time (LoC-Time Protocol)

```nsc
@model(LEDGER, EXEC);

# LoC-Time commit protocol
thread DOMAIN.SCALE.PHASE {
    OBSERVE;
    DECIDE;
    ACT_PHY;
    ACT_CONS;
    AUDIT;
    ACCEPT | ROLLBACK;
    RECEIPT;
}
with require audit, rollback;

@inv(N:INV.clock.stage_coherence, N:INV.ledger.hash_chain_intact);
⇒ (LEDGER, EXEC);
```

**Mandatory Models**: LEDGER, EXEC  
**Required**: Thread structure, Audit/rollback semantics

---

## 8) PIR Opcode Schema

**PIR** (Physics/Proof Intermediate Representation) bridges AST and target models.

### 8.1 Core Opcodes

```python
class OpCode(Enum):
    # Algebraic
    ADD = "add"           # a + b
    MUL = "mul"           # a * b
    COMM = "comm"         # [a, b] (Lie bracket)
    
    # Calculus / Differential
    GRAD = "grad"         # ∇
    DIV = "div"           # div
    CURL = "curl"         # curl
    LAPLACIAN = "laplacian"  # Δ
    DDT = "ddt"           # ∂/∂t
    PARTIAL = "partial"   # ∂_i
    
    # Geometry
    COV_DERIV = "cov"     # ∇^g (requires metric)
    CURVATURE = "curv"    # Riemann tensor
    RICCI = "ricci"       # Ricci tensor/scalar
    HODGE = "hodge"       # * (Hodge star)
    
    # Discrete
    STENCIL = "stencil"   # Finite difference stencil
    QUAD = "quad"         # Quadrature rule
    INTERP = "interp"     # Interpolation
    
    # Ledger / Audit
    INVARIANT_CHECK = "inv_check"  # Check invariant
    GATE_EVAL = "gate_eval"        # Evaluate gate
    RECEIPT = "receipt"            # Emit receipt
    
    # Control
    THREAD = "thread"      # PhaseLoom thread
    IF = "if"              # Conditional
    WHILE = "while"        # Loop
```

### 8.2 Opcode Attributes

```python
@dataclass
class PIRInstruction:
    opcode: OpCode
    operands: List[PIRRef]
    result: Optional[PIRRef]
    
    # Semantic attributes
    effects: Set[Effect]
    regularity: SmoothnessClass
    models_required: Set[Model]
    invariants_consumed: List[InvariantID]
    invariants_produced: List[InvariantID]
    metrics_used: List[MetricID]
    metrics_produced: List[MetricID]
```

---

## 9) Standard Library

### 9.1 ALG Operators

```nsc
# Lie algebra operations
comm(A: LieAlgebra(g), B: LieAlgebra(g)) -> LieAlgebra(g);
ad(X: LieAlgebra(g)) -> Operator(LieAlgebra(g));
exp(X: LieAlgebra(g)) -> GroupElement(g);

# Tensor operations
trace(T: Tensor(1,1)) -> Scalar;
sym(T: Tensor(0,2)) -> Tensor(0,2);
antisym(T: Tensor(0,2)) -> Tensor(0,2);
```

### 9.2 CALC Operators

```nsc
# Differential operators
grad(f: Field[Scalar]) -> Field[Vector];
div(v: Field[Vector]) -> Field[Scalar];
curl(v: Field[Vector]) -> Field[Vector];
laplacian(f: Field[Scalar]) -> Field[Scalar];

# Variational
variation(f: Field[Scalar], u: Field[T]) -> Field[T];
euler_lagrange(L: Functional) -> Equation;
```

### 9.3 GEO Operators

```nsc
# Metric-dependent operators
cov_grad(f: Field[Scalar]) -> Field[Vector];  # Requires metric
cov_deriv(v: Field[Vector]) -> Field[Tensor(1,1)];  # Requires metric
laplace_beltrami(f: Field[Scalar]) -> Field[Scalar];

# Curvature
riemann(RM: Metric) -> Field[Tensor(0,4)];
ricci(RM: Metric) -> Field[Tensor(0,2)];
scalar_curvature(RM: Metric) -> Field[Scalar];
```

---

## 10) Invariant Registry Integration

All invariants must reference the terminology registry:

```nsc
# Example invariant references
@inv(
    N:INV.gr.hamiltonian_constraint,    # Hamiltonian constraint (GR)
    N:INV.gr.momentum_constraint,       # Momentum constraint (GR)
    N:INV.ns.div_free,                  # Divergence-free (NS)
    N:INV.ym.gauss_law,                 # Gauss law (YM)
    N:INV.clock.stage_coherence,        # Stage coherence (LoC-Time)
    N:INV.ledger.hash_chain_intact      # Ledger integrity
)
```

### 10.1 Receipt Field Mapping

Receipts are generated according to the invariant-to-field mappings:

| Invariant ID | Receipt Field | Gate Key |
|--------------|---------------|----------|
| N:INV.gr.hamiltonian_constraint | residuals.eps_H | gates.hamiltonian_constraint |
| N:INV.gr.momentum_constraint | residuals.eps_M | gates.momentum_constraint |
| N:INV.ns.div_free | residuals.eps_div | gates.div_free |
| N:INV.ym.gauss_law | residuals.eps_G | gates.gauss_law |

---

## 11) Compliance Checklist

### 11.1 Syntax Compliance

- [x] All source files parse without errors
- [x] Grammar rules cover all constructs
- [x] Macro expansions produce valid core grammar

### 11.2 Static Semantics Compliance

- [x] Every node has `type` attribute
- [x] Every node has `domains_used` attribute
- [x] Regularity constraints enforced
- [x] Model tags specified for all constructs

### 11.3 Multi-Model Compliance

- [x] Model selection present (`@model`)
- [x] Model projections compile successfully
- [x] Receipt generated for each compilation
- [x] Equivalence tracking for multi-model targets

### 11.4 Dialect Compliance

- [x] Dialect mandatory models present
- [x] Required invariants specified
- [x] Receipt fields match dialect spec
- [x] Operator restrictions respected

---

## 12) NSC-M3L Compiler Upgrade Completion Summary

**Version**: 1.0.1  
**Date**: 2026-01-31  
**Status**: Complete - 59/59 Tests Passing

### Upgrade Summary

| Phase | Component | Changes | Status |
|-------|-----------|---------|--------|
| 1 | Lexer | 18 new physics tokens (DIVERGENCE, CURL, LAPLACIAN, GRAD, TRACE, DET, CONTRACT, VECTOR, SCALAR, SYMMETRIC, ANTISYMMETRIC, etc.) | ✅ Complete |
| 2 | Parser | Physics operator parsing with parentheses syntax (e.g., `div(v)`, `curl(v)`, `grad(phi)`), AST nodes for Divergence, Curl, Gradient, Laplacian, Trace, Determinant, Contraction | ✅ Complete |
| 3 | Type Checker | Physics type checking methods, 10+ intrinsic type signatures for VectorType, SymmetricTensorType, AntiSymmetricTensorType | ✅ Complete |
| 4 | NIR Lowering | Physics operator lowering to NIR, dialect-specific rules (NSC_GR, NSC_NS, NSC_YM) | ✅ Complete |
| 5 | Testing | Full test suite with 59 tests covering all phases | ✅ Complete |

### Supported Physics Operators

| Operator | Syntax | Type Signature | Models |
|----------|--------|----------------|--------|
| Divergence | `div(v)` | Field[Vector] → Field[Scalar] | CALC, GEO |
| Curl | `curl(v)` | Field[Vector] → Field[Vector] | CALC, GEO |
| Gradient | `grad(phi)` | Field[Scalar] → Field[Vector] | CALC, GEO |
| Laplacian | `laplacian(v)` | Field[Vector] → Field[Vector] | CALC, GEO |
| Trace | `trace(T)` | Tensor(n,n) → Scalar | ALG, CALC |
| Determinant | `det(g)` | Tensor(n,n) → Scalar | ALG, CALC |
| Contraction | `contract(T, i, j)` | Tensor(n,n) × indices → Scalar | ALG, CALC |

### Test Coverage

- **Lexer Tests**: 18 tests for physics token recognition
- **Parser Tests**: 11 tests for AST node creation
- **Type Checker Tests**: 12 tests for physics type checking
- **NIR Lowering Tests**: 9 tests for lowering physics constructs
- **Dialect Tests**: 6 tests for NSC_GR, NSC_NS, NSC_YM
- **Integration Tests**: 3 tests for full compilation pipeline

### Dialect-Specific Support

| Dialect | Mandatory Models | Key Invariants | Status |
|---------|------------------|----------------|--------|
| NSC_GR | GEO, CALC, LEDGER, EXEC | hamiltonian_constraint, momentum_constraint | ✅ |
| NSC_NS | CALC, LEDGER, EXEC | div_free, energy_nonincreasing | ✅ |
| NSC_YM | ALG, GEO, CALC, LEDGER, EXEC | gauss_law, bianchi_identity | ✅ |

---

## 13) Related Documents

- [`terminology_registry.json`](terminology_registry.json) — Machine-readable ontology
- [`specifications/aeonica/42_AEONICA_RECEIPTS.md`](specifications/aeonica/42_AEONICA_RECEIPTS.md) — Aeonica receipt spec
- [`plans/nsc_gr_dialect_spec.md`](plans/nsc_gr_dialect_spec.md) — NSC_GR dialect
- [`plans/nsc_ns_dialect_spec.md`](plans/nsc_ns_dialect_spec.md) — NSC_NS dialect
- [`plans/nsc_ym_dialect_spec.md`](plans/nsc_ym_dialect_spec.md) — NSC_YM dialect
- [`plans/nsc_invariant_receipt_spec.md`](plans/nsc_invariant_receipt_spec.md) — Invariant/receipt specs
