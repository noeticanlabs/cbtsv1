from typing import Union, List, Optional
from .ast import Atom, Group, Phrase, Sentence, Program, Span, Node

# Error constants adapted
E_PARSE_EOF = 6
E_PARSE_UNEXPECTED_TOKEN = 7
E_PARSE_UNCLOSED_GROUP = 8
E_PARSE_TRAILING_INPUT = 9
E_PARSE_EMPTY_SENTENCE = 10

class Parser:
    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.pos = 0

    def at_end(self) -> bool:
        return self.pos >= len(self.tokens)

    def current(self) -> str:
        if self.at_end():
            raise ValueError("Unexpected end of input")  # Adapted
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
            raise ValueError("Empty sentence")  # Adapted
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
            raise ValueError(f"Unexpected token: {token}")  # Adapted
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
            raise ValueError(f"Unexpected group start: {open}")  # Adapted
        inner = self.parse_phrase(stop_tokens={close})
        if self.at_end() or self.current() != close:
            raise ValueError(f"Unclosed group, expected {close}")  # Adapted
        self.advance()
        end = self.pos
        return Group(start, end, delim, inner)

def parse_program(tokens: List[str]) -> Program:
    parser = Parser(tokens)
    program = parser.parse_program()
    if not parser.at_end():
        raise ValueError("Trailing input after program")  # Adapted
    return program