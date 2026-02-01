"""
NLLC AST Expression Nodes

Expression types for the NLLC language including physics operators.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from .base import Expr, Node


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
class TypeExpr(Node):
    """Type expression: Vector | Scalar | Tensor(symmetric) | Field[Vector]"""
    name: str           # Base type name (e.g., 'Vector', 'Scalar', 'Tensor')
    modifiers: List[str]  # Type modifiers (e.g., ['symmetric'], ['Field', 'Vector'])


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
