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
    # EOF
    EOF = "EOF"

@dataclass
class Token:
    kind: TokenKind
    value: str
    span: Span

def tokenize(source: str) -> List[Token]:
    tokens = []
    pos = 0
    patterns = [
        (r'let', TokenKind.LET),
        (r'mut', TokenKind.MUT),
        (r'fn', TokenKind.FN),
        (r'if', TokenKind.IF),
        (r'else', TokenKind.ELSE),
        (r'while', TokenKind.WHILE),
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
        (r'[a-zA-Z_][a-zA-Z0-9_]*', TokenKind.IDENT),
        (r'\d*\.\d+([eE][+-]?\d+)?|\d+[eE][+-]?\d+', TokenKind.FLOAT),
        (r'\d+', TokenKind.INT),
        (r'"([^"\\]|\\.)*"', TokenKind.STRING),
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
    ]
    while pos < len(source):
        if source[pos].isspace():
            pos += 1
            continue
        if source[pos:pos+2] == '//':
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