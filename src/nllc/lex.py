import re
from dataclasses import dataclass
from enum import Enum
from typing import List
from .ast import Span

class TokenKind(Enum):
    # Keywords
    LET = "let"
    MUT = "mut"
    FN = "fn"
    IF = "if"
    ELSE = "else"
    WHILE = "while"
    BREAK = "break"
    RETURN = "return"
    IMPORT = "import"
    THREAD = "thread"
    WITH = "with"
    REQUIRE = "require"
    AUDIT = "audit"
    ROLLBACK = "rollback"
    TRUE = "true"
    FALSE = "false"
    CALL = "call"
    # Literals
    INT = "INT"
    FLOAT = "FLOAT"
    STRING = "STRING"
    # Ident
    IDENT = "IDENT"
    # Operators
    ASSIGN = "="
    PLUS = "+"
    MINUS = "-"
    MUL = "*"
    DIV = "/"
    EQ = "=="
    NE = "!="
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="
    AND = "and"
    DOT = "."
    LPAREN = "("
    RPAREN = ")"
    LBRACE = "{"
    RBRACE = "}"
    LBRACKET = "["
    RBRACKET = "]"
    COLON = ":"
    COMMA = ","
    SEMI = ";"
    # NSC-M3L Physics Tokens (Phase 1)
    # Dialect markers
    DIALECT = "dialect"
    # Physics declarations
    FIELD = "field"
    TENSOR = "tensor"
    METRIC = "metric"
    INVARIANT = "invariant"
    GAUGE = "gauge"
    # Physics operators
    DIVERGENCE = "div"
    CURL = "curl"
    LAPLACIAN = "laplacian"
    TRACE = "trace"
    DET = "det"
    CONTRACT = "contract"
    GRAD = "grad"
    # Constraint gates
    CONS = "cons"
    SEM = "sem"
    PHY = "phy"
    VIA = "via"
    # Type keywords
    VECTOR = "vector"
    SCALAR = "scalar"
    SYMMETRIC = "symmetric"
    ANTISYMMETRIC = "antisymmetric"
    # EOF
    EOF = "EOF"

@dataclass
class Token:
    kind: TokenKind
    value: str
    span: Span

class Lexer:
    """Simple lexer wrapper for NLLC source code."""
    
    def __init__(self, source: str):
        self.source = source
    
    def tokenize(self) -> List[Token]:
        """Tokenize the source code."""
        return tokenize(self.source)

def tokenize(source: str) -> List[Token]:
    tokens = []
    pos = 0
    patterns = [
        # NSC-M3L Physics Tokens (Phase 1) - Order matters: longer patterns first
        (r'\bdialect\b', TokenKind.DIALECT),
        (r'\bfield\b', TokenKind.FIELD),
        (r'\btensor\b', TokenKind.TENSOR),
        (r'\bmetric\b', TokenKind.METRIC),
        (r'\binvariant\b', TokenKind.INVARIANT),
        (r'\bgauge\b', TokenKind.GAUGE),
        (r'\bdiv\b', TokenKind.DIVERGENCE),
        (r'\bcurl\b', TokenKind.CURL),
        (r'\blaplacian\b', TokenKind.LAPLACIAN),
        (r'\btrace\b', TokenKind.TRACE),
        (r'\bdet\b', TokenKind.DET),
        (r'\bcontract\b', TokenKind.CONTRACT),
        (r'\bgrad\b', TokenKind.GRAD),
        (r'\bcons\b', TokenKind.CONS),
        (r'\bsem\b', TokenKind.SEM),
        (r'\bphy\b', TokenKind.PHY),
        (r'\bvia\b', TokenKind.VIA),
        (r'\bvector\b', TokenKind.VECTOR),
        (r'\bscalar\b', TokenKind.SCALAR),
        (r'\bsymmetric\b', TokenKind.SYMMETRIC),
        (r'\bantisymmetric\b', TokenKind.ANTISYMMETRIC),
        # Existing keywords
        (r'let', TokenKind.LET),
        (r'mut', TokenKind.MUT),
        (r'fn', TokenKind.FN),
        (r'if', TokenKind.IF),
        (r'else', TokenKind.ELSE),
        (r'while', TokenKind.WHILE),
        (r'break', TokenKind.BREAK),
        (r'return', TokenKind.RETURN),
        (r'import', TokenKind.IMPORT),
        (r'thread', TokenKind.THREAD),
        (r'with', TokenKind.WITH),
        (r'require', TokenKind.REQUIRE),
        (r'audit', TokenKind.AUDIT),
        (r'rollback', TokenKind.ROLLBACK),
        (r'true', TokenKind.TRUE),
        (r'false', TokenKind.FALSE),
        (r'call', TokenKind.CALL),
        # Literals
        (r'\d*\.\d+([eE][+-]?\d+)?|\d+[eE][+-]?\d+', TokenKind.FLOAT),
        (r'\d+', TokenKind.INT),
        (r'"([^"\\]|\\.)*"', TokenKind.STRING),
        # Operators
        (r'\+', TokenKind.PLUS),
        (r'-', TokenKind.MINUS),
        (r'\*', TokenKind.MUL),
        (r'/', TokenKind.DIV),
        (r'==', TokenKind.EQ),
        (r'!=', TokenKind.NE),
        (r'<=', TokenKind.LE),
        (r'>=', TokenKind.GE),
        (r'<', TokenKind.LT),
        (r'>', TokenKind.GT),
        (r'and', TokenKind.AND),
        (r'=', TokenKind.ASSIGN),
        (r'\.', TokenKind.DOT),
        (r'\(', TokenKind.LPAREN),
        (r'\)', TokenKind.RPAREN),
        (r'\{', TokenKind.LBRACE),
        (r'\}', TokenKind.RBRACE),
        (r'\[', TokenKind.LBRACKET),
        (r'\]', TokenKind.RBRACKET),
        (r':', TokenKind.COLON),
        (r',', TokenKind.COMMA),
        (r';', TokenKind.SEMI),
        # Identifiers (must be last)
        (r'[a-zA-Z_][a-zA-Z0-9_]*', TokenKind.IDENT),
    ]
    while pos < len(source):
        if source[pos].isspace():
            pos += 1
            continue
        if source[pos:pos+2] == '//' or source[pos] == '@':
            # Skip comment until end of line
            while pos < len(source) and source[pos] != '\n':
                pos += 1
            if pos < len(source):
                pos += 1  # skip the \n
            continue
        matched = False
        for pat, kind in patterns:
            regex = re.compile(pat)
            match = regex.match(source, pos)
            if match:
                value = match.group(0)
                span = Span(start=pos, end=pos + len(value))
                tokens.append(Token(kind=kind, value=value, span=span))
                pos += len(value)
                matched = True
                break
        if not matched:
            raise ValueError(f"Unexpected character at position {pos}: '{source[pos]}'")
    tokens.append(Token(TokenKind.EOF, "", Span(pos, pos)))
    return tokens
