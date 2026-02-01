"""
NSC-M3L AST Directive Nodes

Directive nodes including model selection, invariants, gates, and compile targets.
"""

from dataclasses import dataclass, field
from typing import Optional, Set, List, Dict, OrderedDict
from .base import Node, Span
from .enums import DirectiveType, Model


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
