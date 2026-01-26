from dataclasses import dataclass
from typing import List, Union
from .flatten import FlatGlyph
from .ast import Span

GLYPH_TO_OPCODE = {
    "φ": 1,
    "↻": 2,
    "⊕": 3,
    "⊖": 4,
    "◯": 5,
    "∆": 6,
    "⇒": 7,
    "□": 8,
    "+": 9,
    "-": 10,
    "*": 11,
    "/": 12,
    "∂": 13,
    "∇": 14,
    "∇²": 15,
    "=": 16,
    "(": 17,
    ")": 18,
    "𝔊": 0x23,
}

@dataclass
class TraceEntry:
    path: str
    span: Span
    file: str = None
    file_sentence: int = None
    module_sentence: int = None

@dataclass
class Bytecode:
    opcodes: list[Union[int, str]]
    trace: list[TraceEntry]

def compile_to_bytecode(flat: list[FlatGlyph], file_path: str = None) -> Bytecode:
    code = []
    trace = []
    for fg in flat:
        glyph = fg.glyph
        if glyph == "⇒":
            opcode = 7
        elif glyph in GLYPH_TO_OPCODE:
            opcode = GLYPH_TO_OPCODE[glyph]
        elif glyph.isalnum() or ('.' in glyph and glyph.replace('.', '').isdigit()):
            opcode = glyph
        else:
            raise ValueError(f"Unknown glyph: {glyph} at {fg.span.start}-{fg.span.end}")
        code.append(opcode)
        # Parse sentence index from path
        parts = fg.path.split('.')
        sentence = int(parts[1]) if parts[0] == 'sentences' and len(parts) > 1 else 0
        file_sentence = sentence
        if file_path is None:
            file = None
            module_sentence = sentence
        else:
            file = file_path
            module_sentence = None  # Will be set later for modules
        trace.append(TraceEntry(path=fg.path, span=fg.span, file=file, file_sentence=file_sentence, module_sentence=module_sentence))
    return Bytecode(opcodes=code, trace=trace)