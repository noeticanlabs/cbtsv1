from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Span:
    start: int
    end: int

@dataclass
class Node:
    span: Span

@dataclass
class Program(Node):
    statements: List['Statement']

# Forward declarations for mutually recursive types
class Statement(Node):
    pass

class Expr(Node):
    pass

@dataclass
class ImportStmt(Statement):
    name: str

@dataclass
class LetStmt(Statement):
    var: str
    expr: Expr

@dataclass
class MutStmt(Statement):
    var: str
    expr: Expr

@dataclass
class AssignStmt(Statement):
    lvalue: Expr
    expr: Expr

@dataclass
class IfStmt(Statement):
    cond: Expr
    body: List[Statement]
    else_body: Optional[List[Statement]] = None

@dataclass
class WhileStmt(Statement):
    cond: Expr
    body: List[Statement]

@dataclass
class ReturnStmt(Statement):
    expr: Optional[Expr] = None

@dataclass
class BreakStmt(Statement):
    pass

@dataclass
class ExprStmt(Statement):
    expr: Expr

@dataclass
class FnDecl(Statement):
    name: str
    params: List[str]
    body: List[Statement]

@dataclass
class ThreadBlock(Statement):
    domain: str
    scale: str
    phase: str
    body: List[Statement]
    require: List[str]

# NSC-M3L Phase 2: Physics AST Nodes

@dataclass
class DialectStmt(Statement):
    """Dialect declaration: dialect ns | gr | ym;"""
    name: str  # 'ns', 'gr', 'ym'

@dataclass
class TypeExpr(Node):
    """Type expression: Vector | Scalar | Tensor(symmetric) | Field[Vector]"""
    name: str           # Base type name (e.g., 'Vector', 'Scalar', 'Tensor')
    modifiers: List[str]  # Type modifiers (e.g., ['symmetric'], ['Field', 'Vector'])

@dataclass
class FieldDecl(Statement):
    """Field declaration: field u: Vector;"""
    name: str           # Field name (e.g., 'u')
    field_type: TypeExpr  # Type expression

@dataclass
class TensorDecl(Statement):
    """Tensor declaration: tensor F: symmetric;"""
    name: str           # Tensor name (e.g., 'F')
    tensor_type: TypeExpr  # Type expression

@dataclass
class MetricDecl(Statement):
    """Metric declaration: metric g: Schwarzschild;"""
    name: str           # Metric name (e.g., 'g')
    metric_type: str    # Metric type (e.g., 'Schwarzschild', 'Minkowski')

@dataclass
class InvariantStmt(Statement):
    """Invariant constraint: invariant div_free with div(u) == 0.0 require cons, sem;"""
    name: str              # Invariant name (e.g., 'div_free')
    constraint: Expr       # Constraint expression
    gates: List[str]       # Required gates: ['cons', 'sem', 'phy']

@dataclass
class GaugeStmt(Statement):
    """Gauge specification: gauge coulomb with div(A) == 0.0 require phy;"""
    name: str              # Gauge name (e.g., 'coulomb')
    condition: Expr        # Gauge condition expression
    gates: List[str]       # Required gates: ['cons', 'phy']

# Expression types

@dataclass
class IntLit(Expr):
    value: int

@dataclass
class FloatLit(Expr):
    value: float

@dataclass
class BoolLit(Expr):
    value: bool

@dataclass
class StrLit(Expr):
    value: str

@dataclass
class ArrayLit(Expr):
    elements: List[Expr]

@dataclass
class ObjectLit(Expr):
    fields: dict

@dataclass
class Var(Expr):
    name: str

@dataclass
class Index(Expr):
    array: Expr
    index: Expr

@dataclass
class Call(Expr):
    func: str
    args: List[Expr]

@dataclass
class BinOp(Expr):
    left: Expr
    op: str
    right: Expr

@dataclass
class IfExpr(Expr):
    cond: Expr
    body: Expr
    else_body: Optional[Expr] = None

# NSC-M3L Phase 2: Physics Operator Expressions

@dataclass
class Divergence(Expr):
    """Divergence operator: div(u)"""
    argument: Expr  # Field/tensor to take divergence of

@dataclass
class Curl(Expr):
    """Curl operator: curl(v)"""
    argument: Expr  # Vector field to take curl of

@dataclass
class Laplacian(Expr):
    """Laplacian operator: laplacian(phi)"""
    argument: Expr  # Scalar/vector field for Laplacian

@dataclass
class Gradient(Expr):
    """Gradient operator: grad(f)"""
    argument: Expr  # Scalar field for gradient

@dataclass
class Trace(Expr):
    """Trace operator: trace(R)"""
    argument: Expr  # Tensor to trace

@dataclass
class Determinant(Expr):
    """Determinant operator: det(g)"""
    argument: Expr  # Matrix/tensor for determinant

@dataclass
class Contraction(Expr):
    """Index contraction operator: contract(g^mu_nu)"""
    argument: Expr     # Tensor to contract
    indices: List[str]  # Index pairs to contract
