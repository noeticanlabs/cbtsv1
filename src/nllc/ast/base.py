"""
NLLC AST Base Classes

Base classes and forward declarations for the NLLC AST.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Span:
    """Source span for error reporting."""
    start: int
    end: int


@dataclass
class Node:
    """Base AST node."""
    span: Span


@dataclass
class Program(Node):
    """Complete NLLC program."""
    statements: List['Statement']


# Forward declarations for mutually recursive types
class Statement(Node):
    """Statement base class."""
    pass


class Expr(Node):
    """Expression base class."""
    pass
