from dataclasses import dataclass
from typing import Union, List, Optional
from collections import OrderedDict
from enum import Enum
import noetica_nsc_phase1.nsc_diag as nsc_diag

class TokenKind(Enum):
    KW_IMPORT = "KW_IMPORT"
    IDENT = "IDENT"

# Error constants
E_PARSE_EOF = 6
E_PARSE_UNEXPECTED_TOKEN = 7
E_PARSE_UNCLOSED_GROUP = 8
E_PARSE_TRAILING_INPUT = 9
E_PARSE_EMPTY_SENTENCE = 10

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

class Parser:
    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.pos = 0

    def at_end(self) -> bool:
        return self.pos >= len(self.tokens)

    def current(self) -> str:
        if self.at_end():
            raise nsc_diag.NSCError(E_PARSE_EOF, "Unexpected end of input")
        return self.tokens[self.pos]

    def advance(self) -> str:
        token = self.current()
        self.pos += 1
        return token

    def peek(self) -> Union[str, None]:
        if self.pos + 1 >= len(self.tokens):
            return None
        return self.tokens[self.pos + 1]

    def parse_program(self) -> Program:
        start = self.pos
        sentences = []
        while not self.at_end():
            sentences.append(self.parse_sentence())
        end = self.pos
        return Program(start, end, sentences)

    def parse_sentence(self) -> Sentence:
        start = self.pos
        if self.at_end():
            raise nsc_diag.NSCError(E_PARSE_EMPTY_SENTENCE, "Empty sentence")
        lhs = self.parse_phrase(stop_tokens={'⇒'})
        arrow = False
        rhs = None
        arrow_span = None
        if not self.at_end() and self.current() == '⇒':
            arrow = True
            i = self.pos
            arrow_span = Span(start=i, end=i+1)
            self.advance()
            rhs = self.parse_phrase()
        end = self.pos
        return Sentence(start, end, lhs, arrow, rhs, arrow_span)

    def parse_phrase(self, stop_tokens: set[str] = set()) -> Phrase:
        start = self.pos
        items = []
        while not self.at_end() and self.current() not in stop_tokens:
            items.append(self.parse_atom_or_group())
        end = self.pos
        return Phrase(start, end, items)

    def parse_atom_or_group(self) -> Union[Atom, Group]:
        if self.current() in ['(', '[']:
            return self.parse_group()
        else:
            return self.parse_atom()

    def parse_atom(self) -> Atom:
        start = self.pos
        token = self.advance()
        if token in [')', ']']:
            raise nsc_diag.NSCError(E_PARSE_UNEXPECTED_TOKEN, f"Unexpected token: {token}")
        end = self.pos
        return Atom(start, end, token)

    def parse_group(self) -> Group:
        start = self.pos
        open = self.advance()
        if open == '(':
            close = ')'
            delim = '()'
        elif open == '[':
            close = ']'
            delim = '[]'
        else:
            raise nsc_diag.NSCError(E_PARSE_UNEXPECTED_TOKEN, f"Unexpected group start: {open}")
        inner = self.parse_phrase(stop_tokens={close})
        if self.at_end() or self.current() != close:
            raise nsc_diag.NSCError(E_PARSE_UNCLOSED_GROUP, f"Unclosed group, expected {close}")
        self.advance()
        end = self.pos
        return Group(start, end, delim, inner)

    def is_operator(self, token: str) -> bool:
        # Operators that are binary/infix
        operators = {'+', '-', '*', '/', '∂', '∇', '∇²', '=', '⊕', '↻', '∆', '◯', '⊖', '⇒', '□', 'ℋ', '𝓜', '𝔊', '𝔇', '𝔅', '𝔄', '𝔯', '𝕋'}
        return token in operators

def parse_program(tokens: List[str]) -> Program:
    parser = Parser(tokens)
    program = parser.parse_program()
    if not parser.at_end():
        raise nsc_diag.NSCError(E_PARSE_TRAILING_INPUT, "Trailing input after program")
    return program

def parse_sentence(tokens: List[str]) -> Sentence:
    parser = Parser(tokens)
    sentence = parser.parse_sentence()
    if not parser.at_end():
        raise nsc_diag.NSCError(E_PARSE_TRAILING_INPUT, "Trailing input after sentence")
    return sentence

def parse_phrase(tokens: List[str]) -> Phrase:
    parser = Parser(tokens)
    phrase = parser.parse_phrase()
    if not parser.at_end():
        raise nsc_diag.NSCError(E_PARSE_TRAILING_INPUT, "Trailing input after phrase")
    return phrase

def parse_group(tokens: List[str]) -> Group:
    parser = Parser(tokens)
    group = parser.parse_group()
    if not parser.at_end():
        raise nsc_diag.NSCError(E_PARSE_TRAILING_INPUT, "Trailing input after group")
    return group

def parse_atom(tokens: List[str]) -> Atom:
    parser = Parser(tokens)
    atom = parser.parse_atom()
    if not parser.at_end():
        raise nsc_diag.NSCError(E_PARSE_TRAILING_INPUT, "Trailing input after atom")
    return atom