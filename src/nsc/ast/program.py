"""
NSC-M3L AST Program Node

Program-level AST node and Statement type union.
"""

from dataclasses import dataclass, field
from typing import Union, List, OrderedDict
from .base import Node
from .statements import Decl, Equation, Functional, Constraint


# Statement type alias - defined here without Directive to avoid circular import
# Directive will be added in __init__.py
StatementBase = Union[Decl, Equation, Functional, Constraint]


@dataclass
class Program(Node):
    """Complete NSC program."""
    statements: List[StatementBase] = field(default_factory=list)

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['statements'] = [s.to_dict() for s in self.statements]
        return d
