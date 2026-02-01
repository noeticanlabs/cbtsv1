"""
NSC-M3L AST Statement Nodes

Statement nodes including declarations, equations, functionals, constraints, and directives.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Set, Dict, OrderedDict
from .base import Node, Span
from .types import Type


# Forward references for circular dependencies
Predicate = Union['Expr', str]


@dataclass
class Meta(Node):
    """Metadata key-value pairs."""
    pairs: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['pairs'] = self.pairs
        return d


@dataclass
class Decl(Node):
    """Variable declaration: Ident :: Type [ ":" Meta ] ";" """
    ident: str = ""
    decl_type: Optional[Type] = None
    meta: Optional[Meta] = None

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['ident'] = self.ident
        if self.decl_type is not None:
            d['type'] = self.decl_type.to_dict()
        if self.meta is not None:
            d['meta'] = self.meta.to_dict()
        return d


@dataclass
class Equation(Node):
    """Equation: Expr = Expr [ ":" Meta ] ";" """
    lhs: Optional['Expr'] = None
    rhs: Optional['Expr'] = None
    meta: Optional[Meta] = None

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        if self.lhs is not None:
            d['lhs'] = self.lhs.to_dict()
        if self.rhs is not None:
            d['rhs'] = self.rhs.to_dict()
        if self.meta is not None:
            d['meta'] = self.meta.to_dict()
        return d


@dataclass
class Binding(Node):
    """Variable binding for functionals."""
    ident: str = ""
    type: Optional[Type] = None

    def to_dict(self) -> OrderedDict:
        d = OrderedDict()
        d['ident'] = self.ident
        if self.type is not None:
            d['type'] = self.type.to_dict()
        return d


@dataclass
class Functional(Node):
    """Functional definition: J(Bindings) := Expr [ ":" Meta ] ";" """
    bindings: List[Binding] = field(default_factory=list)
    expr: Optional['Expr'] = None
    meta: Optional[Meta] = None

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['bindings'] = [b.to_dict() for b in self.bindings]
        if self.expr is not None:
            d['expr'] = self.expr.to_dict()
        if self.meta is not None:
            d['meta'] = self.meta.to_dict()
        return d


@dataclass
class Constraint(Node):
    """Constraint definition: C(Ident) := Predicate [ ":" Meta ] ";" """
    ident: str = ""
    predicate: Optional[Predicate] = None
    meta: Optional[Meta] = None

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['ident'] = self.ident
        if self.predicate is not None:
            if isinstance(self.predicate, str):
                d['predicate'] = self.predicate
            else:
                d['predicate'] = self.predicate.to_dict()
        if self.meta is not None:
            d['meta'] = self.meta.to_dict()
        return d
