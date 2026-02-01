"""
NSC-M3L AST Base Classes

Base classes for all AST nodes with compiler metadata support.
"""

from dataclasses import dataclass, field
from typing import Optional, Set, List, Dict, OrderedDict
from collections import OrderedDict
from .enums import SemanticType, SmoothnessClass, Model


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


@dataclass
class Type(Node):
    """Base type node."""
    pass
