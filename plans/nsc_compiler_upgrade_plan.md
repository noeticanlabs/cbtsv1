# NSC Compiler Upgrade Plan for Dialect Support

**Plan Version**: 1.0  
**Created**: 2026-01-31  
**Objective**: Enable NSC_NS (Navier-Stokes) and NSC_YM (Yang-Mills) dialect compilation

---

## 1. Executive Summary

The current NLLC compiler lacks infrastructure for physics-specific dialects. This plan details phased upgrades to the lexer, parser, and AST to support NSC_NS and NSC_YM dialects with their respective invariants and operators.

### Current Limitations
- All identifiers collapse to single `IDENT` token
- No type annotation system
- No dialect markers for semantic analysis
- Missing physics operators (divergence, curl, trace, etc.)

### Target Capabilities
- Physics-aware tokenization for field/tensor/metric declarations
- Type annotation parsing for vector/tensor/field types
- Invariant constraint declarations with gate requirements
- Gauge specification support (YM dialect)

---

## 2. Phase 1: Lexer Enhancements

### 2.1 New Token Kinds

Add to `src/nllc/lex.py` in `TokenKind` enum:

```python
# Physics declarations
FIELD = "field"          # field u: VectorField
TENSOR = "tensor"        # tensor R: Riemann
METRIC = "metric"        # metric g: Schwarzschild

# Dialect markers
DIALECT = "dialect"      # dialect ns | ym

# Invariants & Constraints
INVARIANT = "invariant"  # invariant div_free with constraint
GAUGE = "gauge"          # gauge coulomb

# Physics operators
DIVERGENCE = "div"       # div F
CURL = "curl"            # curl E
LAPLACIAN = "laplacian"  # laplacian phi
TRACE = "trace"          # trace R
DET = "det"              # det g
CONTRACT = "contract"    # contract indices

# Constraint gates
CONS = "cons"            # Constraint gate
SEM = "sem"              # Semantic gate
PHY = "phy"              # Physical gate

# Gate qualifiers
REQUIRE = "require"      # require sem, cons, phy;
VIA = "via"              # via audit, rollback;

# Type keywords
VECTOR = "vector"        # vector type
SCALAR = "scalar"        # scalar type
SYMMETRIC = "symmetric"  # symmetric tensor
ANTISYMMETRIC = "antisymmetric"  # antisymmetric tensor
```

### 2.2 Lexer Pattern Additions

Add regex patterns for new tokens:

```python
(r'\bfield\b', TokenKind.FIELD),
(r'\btensor\b', TokenKind.TENSOR),
(r'\bmetric\b', TokenKind.METRIC),
(r'\bdialect\b', TokenKind.DIALECT),
(r'\binvariant\b', TokenKind.INVARIANT),
(r'\bgauge\b', TokenKind.GAUGE),
(r'\bdiv\b', TokenKind.DIVERGENCE),
(r'\bcurl\b', TokenKind.CURL),
(r'\blaplacian\b', TokenKind.LAPLACIAN),
(r'\btrace\b', TokenKind.TRACE),
(r'\bdet\b', TokenKind.DET),
(r'\bcontract\b', TokenKind.CONTRACT),
(r'\bcons\b', TokenKind.CONS),
(r'\bsem\b', TokenKind.SEM),
(r'\bphy\b', TokenKind.PHY),
(r'\bvector\b', TokenKind.VECTOR),
(r'\bscalar\b', TokenKind.SCALAR),
(r'\bsymmetric\b', TokenKind.SYMMETRIC),
(r'\bantisymmetric\b', TokenKind.ANTISYMMETRIC),
```

---

## 3. Phase 2: Parser Extensions

### 3.1 New Parsing Methods

Add to `src/nllc/parse.py`:

```python
def parse_dialect(self) -> DialectStmt:
    """Parse dialect declaration: dialect ns | ym;"""
    start = self.current().span.start
    self.expect(TokenKind.DIALECT)
    dialect_tok = self.expect(TokenKind.IDENT)  # ns, ym, gr
    self.expect(TokenKind.SEMI)
    end = self.current().span.end
    return DialectStmt(Span(start, end), dialect_tok.value)

def parse_type_annotation(self) -> TypeExpr:
    """Parse type annotation: ident | vector ident | tensor ident | metric ident"""
    start = self.current().span.start
    
    # Check for type modifiers
    type_modifiers = []
    while self.current().kind in [TokenKind.VECTOR, TokenKind.TENSOR, 
                                   TokenKind.SCALAR, TokenKind.SYMMETRIC,
                                   TokenKind.ANTISYMMETRIC]:
        type_modifiers.append(self.advance().value)
    
    type_name_tok = self.expect(TokenKind.IDENT)
    
    return TypeExpr(
        Span(start, type_name_tok.span.end),
        name=type_name_tok.value,
        modifiers=type_modifiers
    )

def parse_invariant(self) -> InvariantStmt:
    """Parse invariant: invariant NAME with EXPR require GATES;"""
    start = self.current().span.start
    self.expect(TokenKind.INVARIANT)
    name_tok = self.expect(TokenKind.IDENT)
    self.expect(TokenKind.WITH)
    constraint = self.parse_expr()
    self.expect(TokenKind.REQUIRE)
    
    gates = []
    while self.current().kind != TokenKind.SEMI:
        if self.current().kind == TokenKind.CONS:
            gates.append('cons')
            self.advance()
        elif self.current().kind == TokenKind.SEM:
            gates.append('sem')
            self.advance()
        elif self.current().kind == TokenKind.PHY:
            gates.append('phy')
            self.advance()
        else:
            raise ParseError(f"Expected cons, sem, or phy, got {self.current().kind}")
        
        if self.current().kind == TokenKind.COMMA:
            self.advance()
    
    self.expect(TokenKind.SEMI)
    end = self.current().span.end
    
    return InvariantStmt(Span(start, end), name_tok.value, constraint, gates)

def parse_gauge(self) -> GaugeStmt:
    """Parse gauge: gauge NAME with CONDITION require GATES;"""
    start = self.current().span.start
    self.expect(TokenKind.GAUSE)
    name_tok = self.expect(TokenKind.IDENT)
    self.expect(TokenKind.WITH)
    condition = self.parse_expr()
    self.expect(TokenKind.REQUIRE)
    
    gates = []
    while self.current().kind != TokenKind.SEMI:
        if self.current().kind == TokenKind.CONS:
            gates.append('cons')
        elif self.current().kind == TokenKind.PHY:
            gates.append('phy')
        self.advance()
        if self.current().kind == TokenKind.COMMA:
            self.advance()
    
    self.expect(TokenKind.SEMI)
    end = self.current().span.end
    
    return GaugeStmt(Span(start, end), name_tok.value, condition, gates)

def parse_field(self) -> FieldDecl:
    """Parse field: field NAME: TYPE;"""
    start = self.current().span.start
    self.expect(TokenKind.FIELD)
    name_tok = self.expect(TokenKind.IDENT)
    self.expect(TokenKind.COLON)
    field_type = self.parse_type_annotation()
    self.expect(TokenKind.SEMI)
    end = self.current().span.end
    
    return FieldDecl(Span(start, end), name_tok.value, field_type)
```

### 3.2 Updated parse_statement

Add handling for new statement types:

```python
def parse_statement(self) -> Statement:
    tok = self.current()
    # ... existing cases ...
    
    # New cases for Phase 1
    elif tok.kind == TokenKind.DIALECT:
        return self.parse_dialect()
    elif tok.kind == TokenKind.INVARIANT:
        return self.parse_invariant()
    elif tok.kind == TokenKind.GAUGE:
        return self.parse_gauge()
    elif tok.kind == TokenKind.FIELD:
        return self.parse_field()
    elif tok.kind == TokenKind.TENSOR:
        return self.parse_tensor()
    elif tok.kind == TokenKind.METRIC:
        return self.parse_metric()
    
    # ... rest unchanged ...
```

---

## 4. Phase 3: AST Extensions

### 4.1 New AST Nodes

Add to `src/nllc/ast.py`:

```python
# Dialect marker
@dataclass
class DialectStmt(Statement):
    name: str  # 'ns', 'ym', 'gr'

# Type expression
@dataclass
class TypeExpr(Node):
    name: str          # Base type name (e.g., 'VectorField')
    modifiers: List[str]  # ['vector', 'symmetric']

# Invariant constraint
@dataclass
class InvariantStmt(Statement):
    name: str              # Invariant name (e.g., 'div_free')
    constraint: Expr       # Constraint expression
    gates: List[str]       # Required gates: ['cons', 'sem', 'phy']

# Gauge specification
@dataclass
class GaugeStmt(Statement):
    name: str              # Gauge name (e.g., 'coulomb')
    condition: Expr        # Gauge condition expression
    gates: List[str]       # Required gates: ['cons', 'phy']

# Field declaration
@dataclass
class FieldDecl(Statement):
    name: str              # Field name (e.g., 'u')
    field_type: TypeExpr   # Field type annotation

# Tensor declaration
@dataclass
class TensorDecl(Statement):
    name: str              # Tensor name (e.g., 'F')
    tensor_type: TypeExpr  # Tensor type annotation

# Metric declaration
@dataclass
class MetricDecl(Statement):
    name: str              # Metric name (e.g., 'g')
    metric_type: str       # Metric type (e.g., 'Schwarzschild', 'Minkowski')

# Physics operators
@dataclass
class Divergence(Expr):
    argument: Expr         # Field/tensor to take divergence of

@dataclass
class Curl(Expr):
    argument: Expr         # Vector field to take curl of

@dataclass
class Laplacian(Expr):
    argument: Expr         # Scalar/vector field for Laplacian

@dataclass
class Trace(Expr):
    argument: Expr         # Tensor to trace

@dataclass
class Determinant(Expr):
    argument: Expr         # Matrix/tensor for determinant

@dataclass
class Contraction(Expr):
    argument: Expr         # Tensor to contract
    indices: List[str]     # Index pairs to contract
```

### 4.2 Parser Integration for Physics Operators

Update `parse_atom()` to handle physics operators:

```python
def parse_atom(self) -> Expr:
    tok = self.current()
    # ... existing cases ...
    
    # Physics operators
    elif tok.kind == TokenKind.DIVERGENCE:
        start = tok.span.start
        self.advance()
        self.expect(TokenKind.LPAREN)
        arg = self.parse_expr()
        self.expect(TokenKind.RPAREN)
        return Divergence(Span(start, self.current().span.end), arg)
    
    elif tok.kind == TokenKind.CURL:
        start = tok.span.start
        self.advance()
        self.expect(TokenKind.LPAREN)
        arg = self.parse_expr()
        self.expect(TokenKind.RPAREN)
        return Curl(Span(start, self.current().span.end), arg)
    
    elif tok.kind == TokenKind.LAPLACIAN:
        start = tok.span.start
        self.advance()
        self.expect(TokenKind.LPAREN)
        arg = self.parse_expr()
        self.expect(TokenKind.RPAREN)
        return Laplacian(Span(start, self.current().span.end), arg)
    
    elif tok.kind == TokenKind.TRACE:
        start = tok.span.start
        self.advance()
        self.expect(TokenKind.LPAREN)
        arg = self.parse_expr()
        self.expect(TokenKind.RPAREN)
        return Trace(Span(start, self.current().span.end), arg)
    
    elif tok.kind == TokenKind.DET:
        start = tok.span.start
        self.advance()
        self.expect(TokenKind.LPAREN)
        arg = self.parse_expr()
        self.expect(TokenKind.RPAREN)
        return Determinant(Span(start, self.current().span.end), arg)
    
    # ... rest unchanged ...
```

---

## 5. Phase 4: Type Checker Extensions

### 5.1 Type System Additions

Update `src/nllc/type_checker.py`:

```python
# Predefined types
PREDEFINED_TYPES = {
    'Scalar': {'kind': 'scalar'},
    'Vector': {'kind': 'vector'},
    'VectorField': {'kind': 'field', 'base': 'Vector'},
    'Tensor': {'kind': 'tensor'},
    'SymmetricTensor': {'kind': 'tensor', 'symmetric': True},
    'Riemann': {'kind': 'tensor', 'rank': 4},
    'Ricci': {'kind': 'tensor', 'rank': 2},
    'Einstein': {'kind': 'tensor', 'rank': 2},
    'Metric': {'kind': 'metric'},
    'Schwarzschild': {'kind': 'metric'},
    'Minkowski': {'kind': 'metric'},
}

def typecheck_divergence(node: Divergence, ctx: TypeContext) -> TypeResult:
    """div: Field[T] -> Vector"""
    arg_type = typecheck(node.argument, ctx)
    if arg_type.kind == 'field':
        return Type('Vector')
    raise TypeError(f"Cannot take divergence of non-field type {arg_type}")

def typecheck_curl(node: Curl, ctx: TypeContext) -> TypeResult:
    """curl: Vector -> Vector"""
    arg_type = typecheck(node.argument, ctx)
    if arg_type.name == 'Vector':
        return Type('Vector')
    raise TypeError(f"Cannot take curl of non-vector type {arg_type}")
```

### 5.2 Dialect-Aware Type Checking

```python
class DialectAwareChecker:
    def __init__(self, dialect: str):
        self.dialect = dialect
        self.invariants = []
        self.gauges = []
    
    def check_invariant(self, stmt: InvariantStmt) -> TypeResult:
        if self.dialect == 'ns' and stmt.name == 'div_free':
            # Check: div(u) == 0 for incompressible flow
            pass
        elif self.dialect == 'ym' and stmt.name == 'gauss_law':
            # Check: div(E) == rho
            pass
        # ... other invariants
```

---

## 6. Phase 5: Lowering to NIR

### 6.1 Dialect-Specific Lowering

Update `src/nllc/lower_nir.py`:

```python
def lower_invariant(self, stmt: InvariantStmt) -> NIR.Invariant:
    """Lower invariant to NIR constraint."""
    constraint_expr = self.lower_expr(stmt.constraint)
    return NIR.Invariant(
        name=stmt.name,
        constraint=constraint_expr,
        gates=stmt.gates
    )

def lower_gauge(self, stmt: GaugeStmt) -> NIR.Gauge:
    """Lower gauge to NIR gauge condition."""
    condition = self.lower_expr(stmt.condition)
    return NIR.Gauge(
        name=stmt.name,
        condition=condition,
        gates=stmt.gates
    )
```

---

## 7. Example NSC_NS Program

After implementation, NSC_NS programs will look like:

```nllc
dialect ns;

invariant N:INV.pde.div_free with 
    div(field_u) == 0.0
require cons, sem;

invariant N:INV.pde.energy_nonincreasing with
    ddt(energy) <= 0.0
require sem;

field u: Vector;
field p: Scalar;
metric g: Minkowski;

fn compute_divergence(field f: Vector) -> Scalar {
    // Physics kernel call
    return call(divergence_kernel, f);
}

fn main() -> void {
    // Solver loop
    while t < T {
        thread DOMAIN.SCALE.OBSERVE {
            call(sample_fields);
        }
        with require audit, rollback;
    }
}
```

---

## 8. Example NSC_YM Program

```nllc
dialect ym;

gauge N:GAUGE.coulomb with
    div(A) == 0.0
require phy;

invariant N:INV.ym.gauss_law with
    div(E) == charge_density
require cons;

invariant N:INV.ym.bianchi with
    dF - J == 0.0
require cons, sem;

field A: VectorField;
field F: Tensor;

fn main() -> void {
    // Yang-Mills evolution
}
```

---

## 9. Implementation Order

| Phase | Files Modified | Priority | Effort |
|-------|---------------|----------|--------|
| Phase 1 | `src/nllc/lex.py` | High | 1 day |
| Phase 2 | `src/nllc/parse.py` | High | 2 days |
| Phase 3 | `src/nllc/ast.py` | High | 1 day |
| Phase 4 | `src/nllc/type_checker.py` | Medium | 2 days |
| Phase 5 | `src/nllc/lower_nir.py` | Medium | 1 day |

---

## 10. Testing Strategy

### 10.1 Unit Tests

```python
# test_lexer.py
def test_physics_tokens():
    lexer = Lexer("div u; curl E; trace R;")
    tokens = lexer.tokenize()
    assert tokens[0].kind == TokenKind.DIVERGENCE
    assert tokens[2].kind == TokenKind.CURL

# test_parser.py
def test_invariant_parsing():
    source = "invariant div_free with div(u) == 0.0 require cons, sem;"
    parser = Parser(tokenize(source))
    stmt = parser.parse_invariant()
    assert stmt.name == "div_free"
    assert "cons" in stmt.gates
    assert "sem" in stmt.gates
```

### 10.2 Integration Tests

```python
# test_nsc_ns_integration.py
def test_ns_dialect_compilation():
    source = read("test_programs/navier_stokes.nllc")
    result = compile(source, dialect="ns")
    assert result.errors == []
    assert "div_free" in result.invariants

# test_nsc_ym_integration.py
def test_ym_dialect_compilation():
    source = read("test_programs/yang_mills.nllc")
    result = compile(source, dialect="ym")
    assert result.errors == []
    assert "gauss_law" in result.invariants
```

---

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Token collision with user identifiers | Medium | Use `\b` word boundaries in regex |
| Parser ambiguity with new keywords | Low | Reserve keywords, allow escaped identifiers |
| Type system explosion | High | Start with predefined types, defer user-defined types |
| Performance regression | Medium | Benchmark lexer/parser after changes |

---

## 12. Success Criteria

1. ✅ All 18 new token kinds lex correctly
2. ✅ NSC_NS dialect programs parse without errors
3. ✅ NSC_YM dialect programs parse without errors
4. ✅ Type checker validates field/tensor declarations
5. ✅ Invariant constraints lower to NIR correctly
6. ✅ Existing NLLC programs remain compatible
