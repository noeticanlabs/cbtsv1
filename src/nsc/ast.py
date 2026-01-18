from dataclasses import dataclass
from typing import Union, List, Optional
from collections import OrderedDict

@dataclass
class Node:
    start: int
    end: int

    def to_dict(self) -> OrderedDict:
        return OrderedDict(sorted({
            'end': self.end,
            'start': self.start
        }.items()))

@dataclass
class Span:
    start: int
    end: int

    def to_dict(self) -> OrderedDict:
        return OrderedDict(sorted({
            'end': self.end,
            'start': self.start
        }.items()))

PhraseItem = Union['Atom', 'Group']

@dataclass
class Atom(Node):
    value: str

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['value'] = self.value
        return OrderedDict(sorted(d.items()))

@dataclass
class Group(Node):
    delim: str
    inner: 'Phrase'

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['delim'] = self.delim
        d['inner'] = self.inner.to_dict()
        return OrderedDict(sorted(d.items()))

@dataclass
class Phrase(Node):
    items: List[PhraseItem]

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['items'] = [item.to_dict() for item in self.items]
        return OrderedDict(sorted(d.items()))

@dataclass
class Sentence(Node):
    lhs: Phrase
    arrow: bool
    rhs: Optional[Phrase]
    arrow_span: Optional[Span] = None

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['arrow'] = self.arrow
        d['lhs'] = self.lhs.to_dict()
        d['rhs'] = self.rhs.to_dict() if self.rhs is not None else None
        if self.arrow_span is not None:
            d['arrow_span'] = self.arrow_span.to_dict()
        return OrderedDict(sorted(d.items()))

@dataclass
class Program(Node):
    sentences: List[Sentence]

    def to_dict(self) -> OrderedDict:
        d = super().to_dict()
        d['sentences'] = [s.to_dict() for s in self.sentences]
        return OrderedDict(sorted(d.items()))