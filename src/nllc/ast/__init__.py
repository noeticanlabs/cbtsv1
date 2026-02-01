"""
NLLC Abstract Syntax Tree

This module has been modularized into submodules:
- base: Span, Node, Program, Statement, Expr
- statements: ImportStmt, LetStmt, AssignStmt, IfStmt, etc.
- expressions: IntLit, FloatLit, Var, Call, BinOp, etc.

For backward compatibility, all exports are also available from this module.
"""

from .base import (
    Span,
    Node,
    Program,
    Statement,
    Expr
)

from .statements import (
    ImportStmt,
    LetStmt,
    MutStmt,
    AssignStmt,
    IfStmt,
    WhileStmt,
    ReturnStmt,
    BreakStmt,
    ExprStmt,
    FnDecl,
    ThreadBlock,
    DialectStmt,
    FieldDecl,
    TensorDecl,
    MetricDecl,
    InvariantStmt,
    GaugeStmt
)

from .expressions import (
    IntLit,
    FloatLit,
    BoolLit,
    StrLit,
    ArrayLit,
    ObjectLit,
    Var,
    Index,
    Call,
    BinOp,
    IfExpr,
    TypeExpr,
    Divergence,
    Curl,
    Laplacian,
    Gradient,
    Trace,
    Determinant,
    Contraction
)

__all__ = [
    # Base
    'Span',
    'Node',
    'Program',
    'Statement',
    'Expr',
    # Statements
    'ImportStmt',
    'LetStmt',
    'MutStmt',
    'AssignStmt',
    'IfStmt',
    'WhileStmt',
    'ReturnStmt',
    'BreakStmt',
    'ExprStmt',
    'FnDecl',
    'ThreadBlock',
    'DialectStmt',
    'FieldDecl',
    'TensorDecl',
    'MetricDecl',
    'InvariantStmt',
    'GaugeStmt',
    # Expressions
    'IntLit',
    'FloatLit',
    'BoolLit',
    'StrLit',
    'ArrayLit',
    'ObjectLit',
    'Var',
    'Index',
    'Call',
    'BinOp',
    'IfExpr',
    'TypeExpr',
    'Divergence',
    'Curl',
    'Laplacian',
    'Gradient',
    'Trace',
    'Determinant',
    'Contraction'
]
