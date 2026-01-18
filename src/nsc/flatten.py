from dataclasses import dataclass
from typing import List
from .ast import Program, Phrase, Atom, Group, Span

@dataclass
class FlatGlyph:
    glyph: str
    path: str
    span: Span

def flatten_phrase_to_list(phrase) -> list[str]:
    result = []
    for item in phrase.items:
        if hasattr(item, 'value'):  # Atom
            result.append(item.value)
        elif hasattr(item, 'delim'):  # Group
            result.extend(flatten_phrase_to_list(item.inner))
    return result

def flatten_program(prog: Program) -> list[str]:
    result = []
    for sentence in prog.sentences:
        result.extend(flatten_phrase_to_list(sentence.lhs))
        if sentence.arrow:
            result.append('⇒')
            if sentence.rhs:
                result.extend(flatten_phrase_to_list(sentence.rhs))
    return result

def flatten_phrase(phrase: Phrase, path_prefix: str) -> list[FlatGlyph]:
    result = []
    for i, item in enumerate(phrase.items):
        if isinstance(item, Atom):
            path = f"{path_prefix}.items.{i}"
            result.append(FlatGlyph(glyph=item.value, path=path, span=Span(start=item.start, end=item.end)))
        elif isinstance(item, Group):
            # recurse without delimiters
            result.extend(flatten_phrase(item.inner, f"{path_prefix}.items.{i}.inner"))
    return result

def flatten_with_trace(prog: Program) -> list[FlatGlyph]:
    result = []
    for i, sentence in enumerate(prog.sentences):
        # flatten lhs
        result.extend(flatten_phrase(sentence.lhs, f"sentences.{i}.lhs"))
        if sentence.arrow:
            # insert FlatGlyph for ⇒ with arrow_span
            path = f"sentences.{i}"
            result.append(FlatGlyph(glyph="⇒", path=path, span=sentence.arrow_span))
            # flatten rhs if present
            if sentence.rhs:
                result.extend(flatten_phrase(sentence.rhs, f"sentences.{i}.rhs"))
    return result