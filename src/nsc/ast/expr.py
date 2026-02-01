"""
NSC-M3L AST Expression Nodes

Expression nodes including atoms, groups, operations, and type unions.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, OrderedDict
from .base import Node


# Forward reference for circular dependency
Expr = Union['Atom', 'Group', 'OpCall', 'BinaryOp']


@dataclass
class Atom(Node):
    """Atomic expression: identifier, number, tensor, or field access."""
    value: str = ""

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['value'] = self.value
        return d


@dataclass
class Group(Node):
    """Grouped expression with delimiters."""
    delim: str = ""
    inner: Optional[Expr] = None

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['delim'] = self.delim
        if self.inner is not None:
            d['inner'] = self.inner.to_dict()
        return d


@dataclass
class OpCall(Node):
    """Operator call: Op(Expr) syntax."""
    op: str = ""
    arg: Optional[Expr] = None

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['op'] = self.op
        if self.arg is not None:
            d['arg'] = self.arg.to_dict()
        return d


@dataclass
class BinaryOp(Node):
    """Binary operation."""
    op: str = ""
    left: Optional[Expr] = None
    right: Optional[Expr] = None

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['op'] = self.op
        if self.left is not None:
            d['left'] = self.left.to_dict()
        if self.right is not None:
            d['right'] = self.right.to_dict()
        return d
