"""
NSC-M3L AST Type Nodes

Type definitions including scalars, vectors, tensors, fields, forms, operators, etc.
"""

from dataclasses import dataclass, field
from typing import Optional, OrderedDict
from .base import Node, Type


@dataclass
class ScalarType(Type):
    """Scalar type."""
    pass


@dataclass
class VectorType(Type):
    """Vector type with optional dimension."""
    dim: Optional[int] = None

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        if self.dim is not None:
            d['dim'] = self.dim
        return d


@dataclass
class TensorType(Type):
    """Tensor type with (k, l) valence."""
    k: int = 0
    l: int = 0

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['k'] = self.k
        d['l'] = self.l
        return d


@dataclass
class FieldType(Type):
    """Field type with value type."""
    value_type: Optional[Type] = None

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        if self.value_type is not None:
            d['value_type'] = self.value_type.to_dict()
        return d


@dataclass
class FormType(Type):
    """Differential form type."""
    p: int = 0

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['p'] = self.p
        return d


@dataclass
class OperatorType(Type):
    """Operator type: Domain -> Codomain."""
    domain: Optional[Type] = None
    codomain: Optional[Type] = None

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        if self.domain is not None:
            d['domain'] = self.domain.to_dict()
        if self.codomain is not None:
            d['codomain'] = self.codomain.to_dict()
        return d


@dataclass
class ManifoldType(Type):
    """Manifold type: Manifold(dim, signature)."""
    dim: int = 0
    signature: Optional[str] = None

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['dim'] = self.dim
        if self.signature is not None:
            d['signature'] = self.signature
        return d


@dataclass
class MetricType(Type):
    """Metric type: Metric(dim, signature)."""
    dim: Optional[int] = None
    signature: Optional[str] = None

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        if self.dim is not None:
            d['dim'] = self.dim
        if self.signature is not None:
            d['signature'] = self.signature
        return d


@dataclass
class LieAlgebraType(Type):
    """Lie algebra type: LieAlgebra(name)."""
    name: str = ""

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['name'] = self.name
        return d


@dataclass
class ConnectionType(Type):
    """Connection type."""
    metric_compatible: bool = False
    torsion_free: bool = True

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['metric_compatible'] = self.metric_compatible
        d['torsion_free'] = self.torsion_free
        return d
