"""
NLLC AST Statement Nodes

Statement types for the NLLC language.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from .base import Statement, Expr, Node


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
class FieldDecl(Statement):
    """Field declaration: field u: Vector;"""
    name: str           # Field name (e.g., 'u')
    field_type: 'TypeExpr'  # Type expression


@dataclass
class TensorDecl(Statement):
    """Tensor declaration: tensor F: symmetric;"""
    name: str           # Tensor name (e.g., 'F')
    tensor_type: 'TypeExpr'  # Type expression


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
