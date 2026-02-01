"""
NSC-M3L Enum Definitions

Semantic types, models, smoothness classes, and directive types for the AST.
"""

from enum import Enum


class SemanticType(Enum):
    """Semantic types for NSC-M3L nodes."""
    SCALAR = "Scalar"
    VECTOR = "Vector"
    TENSOR = "Tensor"
    FIELD = "Field"
    FORM = "Form"
    OPERATOR = "Operator"
    FUNCTIONAL = "Functional"
    MANIFOLD = "Manifold"
    METRIC = "Metric"
    CONNECTION = "Connection"
    LIE_ALGEBRA = "LieAlgebra"


class Model(Enum):
    """Semantic models for multi-model compilation."""
    ALG = "ALG"
    CALC = "CALC"
    GEO = "GEO"
    DISC = "DISC"
    LEDGER = "LEDGER"
    EXEC = "EXEC"


class SmoothnessClass(Enum):
    """Smoothness classes for regularity constraints."""
    C0 = "C0"
    C1 = "C1"
    C2 = "C2"
    H1 = "H1"
    H2 = "H2"
    L2 = "L2"


class DirectiveType(Enum):
    """Types of directives."""
    MODEL = "model"
    INV = "inv"
    GATE = "gate"
    COMPILE = "compile"
