from dataclasses import dataclass, field
from typing import Union, List, Optional, Set, Dict
from collections import OrderedDict
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


# === Base Classes ===

@dataclass
class Span:
    """Source span for error reporting."""
    start: int
    end: int

    def to_dict(self) -> OrderedDict:
        return OrderedDict(sorted({
            'end': self.end,
            'start': self.start
        }.items()))


@dataclass
class Node:
    """Base AST node with compiler metadata."""
    start: int
    end: int
    type: Optional[SemanticType] = None
    domains_used: Set[Model] = field(default_factory=set)
    units: Optional[str] = None
    regularity: Optional[SmoothnessClass] = None
    invariants_required: List[str] = field(default_factory=list)
    effects: Set[str] = field(default_factory=set)
    model_tags: Dict[Model, str] = field(default_factory=dict)

    def to_dict(self) -> OrderedDict:
        d = OrderedDict(sorted({
            'end': self.end,
            'start': self.start
        }.items()))
        if self.type is not None:
            d['type'] = self.type.value
        if self.domains_used:
            d['domains_used'] = sorted([m.value for m in self.domains_used])
        if self.units is not None:
            d['units'] = self.units
        if self.regularity is not None:
            d['regularity'] = self.regularity.value
        if self.invariants_required:
            d['invariants_required'] = self.invariants_required
        if self.effects:
            d['effects'] = sorted(self.effects)
        if self.model_tags:
            d['model_tags'] = {k.value: v for k, v in self.model_tags.items()}
        return d


# === Expression Nodes ===

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
    inner: Optional['Expr'] = None

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
    arg: Optional['Expr'] = None

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['op'] = self.op
        if self.arg is not None:
            d['arg'] = self.arg.to_dict()
        return d


Expr = Union[Atom, Group, OpCall, 'BinaryOp']


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


# === Type Nodes ===

@dataclass
class Type(Node):
    """Base type node."""
    pass


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


# === Metadata Nodes ===

@dataclass
class Meta(Node):
    """Metadata key-value pairs."""
    pairs: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['pairs'] = self.pairs
        return d


# === Statement Nodes ===

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
    lhs: Optional[Expr] = None
    rhs: Optional[Expr] = None
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
    expr: Optional[Expr] = None
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
    predicate: Optional['Predicate'] = None
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


Predicate = Union[Expr, str]


# === Directive Nodes ===

@dataclass
class ModelSelector(Node):
    """Model selection directive: @model(ModelList) ";" """
    models: Set[Model] = field(default_factory=set)

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['models'] = sorted([m.value for m in self.models])
        return d


@dataclass
class InvariantList(Node):
    """Invariant list for @inv directive."""
    invariants: List[str] = field(default_factory=list)

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['invariants'] = self.invariants
        return d


@dataclass
class GateSpec(Node):
    """Gate specification for @gate directive."""
    config: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['config'] = self.config
        return d


@dataclass
class TargetList(Node):
    """Target list for ⇒ directive."""
    targets: Set[Model] = field(default_factory=set)

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['targets'] = sorted([m.value for m in self.targets])
        return d


@dataclass
class Directive(Node):
    """Directive statement."""
    directive_type: Optional[DirectiveType] = None
    # Union of all directive content types
    model_selector: Optional[ModelSelector] = None
    invariant_list: Optional[InvariantList] = None
    gate_spec: Optional[GateSpec] = None
    target_list: Optional[TargetList] = None

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        if self.directive_type is not None:
            d['directive_type'] = self.directive_type.value
        if self.model_selector is not None:
            d['models'] = self.model_selector.to_dict()
        if self.invariant_list is not None:
            d['invariants'] = self.invariant_list.to_dict()
        if self.gate_spec is not None:
            d['gate'] = self.gate_spec.to_dict()
        if self.target_list is not None:
            d['targets'] = self.target_list.to_dict()
        return d


# === Statement Union and Program ===

Statement = Union[Decl, Equation, Functional, Constraint, Directive]


@dataclass
class Program(Node):
    """Complete NSC program."""
    statements: List[Statement] = field(default_factory=list)

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['statements'] = [s.to_dict() for s in self.statements]
        return d
