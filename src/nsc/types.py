"""
NSC-M3L Semantic Types

Defines the semantic type system per specifications/nsc_m3l_v1.md section 3.1.
"""

from dataclasses import dataclass, field
from typing import Union, List, Optional, Set, Dict
from enum import Enum


class Dimension(Enum):
    """Physical dimension tags for unit checking."""
    TIME = "time"
    LENGTH = "length"
    MASS = "mass"
    CHARGE = "charge"
    TEMPERATURE = "temperature"
    DIMENSIONLESS = "dimensionless"


class TimeMode(Enum):
    """Temporal semantics for expressions."""
    PHYSICAL = "physical"  # Physical time evolution
    AUDIT = "audit"  # Audit/temporal logging
    BOTH = "both"  # Both physical and audit


class Effect(Enum):
    """Side effects for effect tracking."""
    READ_STATE = "read_state"
    WRITE_STATE = "write_state"
    NONLOCAL = "nonlocal"
    GAUGE_CHANGE = "gauge_change"


class Tag(Enum):
    """Per-model semantic annotation tags."""
    # ALG tags
    COMMUTATIVE = "commutative"
    ASSOCIATIVE = "associative"
    LIE_ALGEBRA = "lie_algebra"
    
    # CALC tags
    DIFFERENTIABLE = "differentiable"
    INTEGRABLE = "integrable"
    SMOOTH = "smooth"
    
    # GEO tags
    METRIC_COMPATIBLE = "metric_compatible"
    TORSION_FREE = "torsion_free"
    LEVI_CIVITA = "levi_civita"
    
    # DISC tags
    STABLE = "stable"
    CONSERVATIVE = "conservative"
    CONSISTENT = "consistent"
    
    # LEDGER tags
    VERIFIED = "verified"
    AUDITED = "audited"
    CERTIFIED = "certified"


# === Base Semantic Types ===

@dataclass
class Scalar:
    """ℝ^n element - base scalar type."""
    dimension: Optional[Dimension] = None
    
    def __str__(self) -> str:
        if self.dimension:
            return f"Scalar[{self.dimension.value}]"
        return "Scalar"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Scalar)


@dataclass
class Vector:
    """Tangent space element - base vector type."""
    dim: Optional[int] = None  # Optional dimension
    
    def __str__(self) -> str:
        if self.dim:
            return f"Vector[{self.dim}]"
        return "Vector"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Vector) and self.dim == other.dim


@dataclass
class Tensor:
    """(k, l) tensor type - k contravariant, l covariant indices."""
    k: int = 0  # Contravariant indices
    l: int = 0  # Covariant indices
    
    def __str__(self) -> str:
        return f"Tensor({self.k},{self.l})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Tensor) and self.k == other.k and self.l == other.l


@dataclass
class Operator:
    """Bounded linear operator: Domain → Codomain."""
    domain: 'SemanticType'
    codomain: 'SemanticType'
    
    def __str__(self) -> str:
        return f"Operator({self.domain} → {self.codomain})"
    
    def __eq__(self, other) -> bool:
        return (isinstance(other, Operator) and 
                self.domain == other.domain and 
                self.codomain == other.codomain)


@dataclass
class Functional:
    """ℝ-valued operator - functional (scalar-valued operator)."""
    domain: 'SemanticType'
    
    def __str__(self) -> str:
        return f"Functional({self.domain} → ℝ)"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Functional) and self.domain == other.domain


@dataclass
class Field:
    """Field over spacetime with values of type T."""
    value_type: 'SemanticType'
    
    def __str__(self) -> str:
        return f"Field[{self.value_type}]"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Field) and self.value_type == other.value_type


@dataclass
class Form:
    """Differential p-form."""
    p: int = 0  # Degree of form
    
    def __str__(self) -> str:
        return f"Form[{self.p}]"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Form) and self.p == other.p


@dataclass
class BundleConnection:
    """Connection on principal bundle for gauge theories."""
    gauge_group: str  # e.g., "SU(N)", "SO(3)"
    
    def __str__(self) -> str:
        return f"BundleConnection({self.gauge_group})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, BundleConnection) and self.gauge_group == other.gauge_group


@dataclass
class Metric:
    """Riemannian or Pseudo-Riemannian metric."""
    signature: Optional[str] = None  # e.g., "+---" for Lorentzian
    dim: Optional[int] = None
    
    def __str__(self) -> str:
        if self.signature and self.dim:
            return f"Metric({self.dim},{self.signature})"
        elif self.dim:
            return f"Metric({self.dim})"
        return "Metric"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Metric) and self.signature == other.signature and self.dim == other.dim


@dataclass
class Manifold:
    """Manifold with dimension and signature."""
    dim: int
    signature: Optional[str] = None  # e.g., "riemannian", "lorentzian"
    
    def __str__(self) -> str:
        if self.signature:
            return f"Manifold({self.dim},{self.signature})"
        return f"Manifold({self.dim})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Manifold) and self.dim == other.dim and self.signature == other.signature


@dataclass
class LieAlgebra:
    """Lie algebra for gauge groups."""
    name: str  # e.g., "su(N)", "so(3)", "u(1)"
    
    def __str__(self) -> str:
        return f"LieAlgebra({self.name})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, LieAlgebra) and self.name == other.name


@dataclass
class Connection:
    """Affine connection on manifold."""
    metric_compatible: bool = False
    torsion_free: bool = True
    
    def __str__(self) -> str:
        flags = []
        if self.metric_compatible:
            flags.append("metric_compatible")
        if self.torsion_free:
            flags.append("torsion_free")
        if flags:
            return f"Connection({', '.join(flags)})"
        return "Connection"
    
    def __eq__(self, other) -> bool:
        return (isinstance(other, Connection) and 
                self.metric_compatible == other.metric_compatible and 
                self.torsion_free == other.torsion_free)


# Union type for all semantic types
SemanticType = Union[
    Scalar, Vector, Tensor, Operator, Functional, Field, 
    Form, BundleConnection, Metric, Manifold, LieAlgebra, Connection
]


# === Type Utilities ===

def get_base_type_name(t: SemanticType) -> str:
    """Get the base type name as a string."""
    return type(t).__name__


def is_numeric(t: SemanticType) -> bool:
    """Check if type is numeric (scalar or tensor)."""
    return isinstance(t, (Scalar, Vector, Tensor))


def is_field_type(t: SemanticType) -> bool:
    """Check if type is a field type."""
    return isinstance(t, Field)


def get_field_value_type(t: Field) -> SemanticType:
    """Extract value type from field."""
    return t.value_type


def compatible_types(a: SemanticType, b: SemanticType) -> bool:
    """Check if two types are compatible for operations."""
    if a == b:
        return True
    # Scalar is compatible with components
    if isinstance(a, Scalar) and isinstance(b, (Scalar, Vector, Tensor)):
        return True
    if isinstance(b, Scalar) and isinstance(a, (Scalar, Vector, Tensor)):
        return True
    return False


# === Type Error ===

class TypeError(Exception):
    """Type checking error with location information."""
    
    def __init__(self, message: str, node_id: Optional[str] = None, 
                 expected: Optional[SemanticType] = None,
                 found: Optional[SemanticType] = None,
                 location: Optional[str] = None):
        self.message = message
        self.node_id = node_id
        self.expected = expected
        self.found = found
        self.location = location
        
        parts = [message]
        if expected:
            parts.append(f"  Expected: {expected}")
        if found:
            parts.append(f"  Found: {found}")
        if location:
            parts.append(f"  Location: {location}")
        
        super().__init__("\n".join(parts))


class GeometryPrerequisiteError(TypeError):
    """Error for missing geometry prerequisites."""
    
    def __init__(self, operator: str, required: str, location: Optional[str] = None):
        super().__init__(
            f"Geometry prerequisite not satisfied for operator '{operator}'",
            expected=required,
            location=location
        )
        self.operator = operator
        self.required = required


class RegularityError(TypeError):
    """Error for regularity constraint violation."""
    
    def __init__(self, operator: str, required: str, found: Optional[str] = None, 
                 location: Optional[str] = None):
        super().__init__(
            f"Regularity constraint violated for operator '{operator}'",
            expected=required,
            found=found,
            location=location
        )
        self.operator = operator
        self.required = required


class ModelCompatibilityError(TypeError):
    """Error for model compatibility violation."""
    
    def __init__(self, expression: str, required_models: Set[str], 
                 found_models: Set[str], location: Optional[str] = None):
        super().__init__(
            f"Model compatibility violation for expression '{expression}'",
            expected=required_models,
            found=found_models,
            location=location
        )
        self.expression = expression
        self.required_models = required_models
        self.found_models = found_models
