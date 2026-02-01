"""
NSC-M3L Abstract Syntax Tree

This module has been modularized. The following submodules are available:
- enums: SemanticType, Model, SmoothnessClass, DirectiveType
- base: Span, Node, Type
- expr: Atom, Group, OpCall, BinaryOp, Expr
- types: ScalarType, VectorType, TensorType, FieldType, FormType, OperatorType, ManifoldType, MetricType, LieAlgebraType, ConnectionType
- statements: Meta, Decl, Equation, Binding, Functional, Constraint, Predicate
- directives: ModelSelector, InvariantList, GateSpec, TargetList, Directive
- program: Program

For backward compatibility, all exports are also available from this module.
"""

from typing import Union

# Re-export enums
from .enums import (
    SemanticType,
    Model,
    SmoothnessClass,
    DirectiveType
)

# Re-export base classes
from .base import (
    Span,
    Node,
    Type
)

# Re-export expression nodes
from .expr import (
    Atom,
    Group,
    OpCall,
    BinaryOp,
    Expr
)

# Re-export type nodes
from .types import (
    ScalarType,
    VectorType,
    TensorType,
    FieldType,
    FormType,
    OperatorType,
    ManifoldType,
    MetricType,
    LieAlgebraType,
    ConnectionType
)

# Re-export statement nodes
from .statements import (
    Meta,
    Decl,
    Equation,
    Binding,
    Functional,
    Constraint,
    Predicate
)

# Re-export directive nodes
from .directives import (
    ModelSelector,
    InvariantList,
    GateSpec,
    TargetList,
    Directive
)

# Re-export Program (Statement is defined below to avoid circular import)
from .program import Program


# Define Statement type union AFTER all imports to avoid circular dependency
# Statement = Union of all statement types including Directive
Statement = Union[Decl, Equation, Functional, Constraint, Directive]


__all__ = [
    # Enums
    'SemanticType',
    'Model',
    'SmoothnessClass',
    'DirectiveType',
    # Base
    'Span',
    'Node',
    'Type',
    # Expressions
    'Atom',
    'Group',
    'OpCall',
    'BinaryOp',
    'Expr',
    # Types
    'ScalarType',
    'VectorType',
    'TensorType',
    'FieldType',
    'FormType',
    'OperatorType',
    'ManifoldType',
    'MetricType',
    'LieAlgebraType',
    'ConnectionType',
    # Statements
    'Meta',
    'Decl',
    'Equation',
    'Binding',
    'Functional',
    'Constraint',
    'Predicate',
    # Directives
    'ModelSelector',
    'InvariantList',
    'GateSpec',
    'TargetList',
    'Directive',
    # Program
    'Program',
    'Statement'
]
