import re
import unicodedata

# Token type constants
TOK_IDENT = 'TOK_IDENT'
TOK_NUMBER = 'TOK_NUMBER'
TOK_COLON_COLON = 'TOK_COLON_COLON'  # ::
TOK_COLON = 'TOK_COLON'  # :
TOK_SEMICOLON = 'TOK_SEMICOLON'  # ;
TOK_ASSIGN = 'TOK_ASSIGN'  # :=
TOK_COMMA = 'TOK_COMMA'  # ,
TOK_LPAREN = 'TOK_LPAREN'  # (
TOK_RPAREN = 'TOK_RPAREN'  # )
TOK_LBRACKET = 'TOK_LBRACKET'  # [
TOK_RBRACKET = 'TOK_RBRACKET'  # ]
TOK_EQUAL = 'TOK_EQUAL'  # =
TOK_PLUS = 'TOK_PLUS'  # +
TOK_MINUS = 'TOK_MINUS'  # -
TOK_STAR = 'TOK_STAR'  # *
TOK_SLASH = 'TOK_SLASH'  # /
TOK_ARROW = 'TOK_ARROW'  # ⇒

# Keywords
KW_MODEL = '@model'
KW_INV = '@inv'
KW_GATE = '@gate'
KW_J = 'J'  # Functional marker

# Model names
MODELS = {'ALG', 'CALC', 'GEO', 'DISC', 'LEDGER', 'EXEC'}

# Type names
TYPES = {'Scalar', 'Vector', 'Tensor', 'Field', 'Form', 'Operator', 'Manifold', 'Metric', 'LieAlgebra', 'Connection'}

# Physics operators
OPERATORS = {'div', 'curl', 'grad', 'laplacian', 'trace', 'det', 'contract'}

def tokenize(nsc: str) -> list[tuple[str, str]]:
    """
    Tokenize NSC source code into tokens with their types.
    
    Returns a list of (token_type, token_value) tuples.
    """
    normalized = unicodedata.normalize('NFC', nsc)
    if normalized != nsc:
        raise ValueError("Non-canonical Unicode")
    
    # Pattern to match operators, identifiers, numbers, and special tokens
    # Order matters: longer tokens should come first
    pattern = r'''
        (@model|@inv|@gate)     |  # Directive keywords
        (⇒)                     |  # Arrow
        (::|:=)                 |  # Double colon or assign
        ([A-Za-z_][A-Za-z0-9_]*) |  # Identifiers
        (\d+\.?\d*)             |  # Numbers
        ([+\-*/=;,()\[\]])      |  # Single-char operators
        (\s+)                   |  # Whitespace (skip)
        (.)                     |  # Anything else (glyphs, etc.)
    '''
    
    tokens = []
    for match in re.finditer(pattern, normalized, re.VERBOSE):
        groups = match.groups()
        
        # Find which group matched (first non-None)
        for i, g in enumerate(groups):
            if g is not None and i < 7:  # Skip whitespace and fallback
                break
        else:
            continue
        
        # Classify the token
        token = g
        if i == 0:  # Directive keywords
            if token == '@model':
                tokens.append(('KW_MODEL', token))
            elif token == '@inv':
                tokens.append(('KW_INV', token))
            elif token == '@gate':
                tokens.append(('KW_GATE', token))
        elif i == 1:  # Arrow
            tokens.append((TOK_ARROW, token))
        elif i == 2:  # Double char operators
            if token == '::':
                tokens.append((TOK_COLON_COLON, token))
            elif token == ':=':
                tokens.append((TOK_ASSIGN, token))
        elif i == 3:  # Identifier
            if token in MODELS:
                tokens.append(('MODEL', token))
            elif token in TYPES:
                tokens.append(('TYPE', token))
            elif token in OPERATORS:
                tokens.append(('OP', token))
            elif token == KW_J:
                tokens.append(('KW_J', token))
            else:
                tokens.append((TOK_IDENT, token))
        elif i == 4:  # Number
            tokens.append((TOK_NUMBER, token))
        elif i == 5:  # Single char operators
            if token == '+':
                tokens.append((TOK_PLUS, token))
            elif token == '-':
                tokens.append((TOK_MINUS, token))
            elif token == '*':
                tokens.append((TOK_STAR, token))
            elif token == '/':
                tokens.append((TOK_SLASH, token))
            elif token == '=':
                tokens.append((TOK_EQUAL, token))
            elif token == ';':
                tokens.append((TOK_SEMICOLON, token))
            elif token == ',':
                tokens.append((TOK_COMMA, token))
            elif token == '(':
                tokens.append((TOK_LPAREN, token))
            elif token == ')':
                tokens.append((TOK_RPAREN, token))
            elif token == '[':
                tokens.append((TOK_LBRACKET, token))
            elif token == ']':
                tokens.append((TOK_RBRACKET, token))
            elif token == ':':
                tokens.append((TOK_COLON, token))
        elif i == 7:  # Glyph or other special character
            tokens.append(('GLYPH', token))
    
    if not tokens:
        raise ValueError("Empty input string")
    
    return tokens


def tokenize_simple(nsc: str) -> list[str]:
    """
    Simple tokenization for backward compatibility.
    Returns just the token values without types.
    """
    tokens = tokenize(nsc)
    return [t[1] for t in tokens]


def tokenize_with_positions(nsc: str) -> list[dict]:
    """
    Tokenize with position information for error reporting.
    
    Returns a list of dicts with keys: type, value, start, end
    """
    normalized = unicodedata.normalize('NFC', nsc)
    
    tokens = []
    pos = 0
    
    # Pattern for tokenization
    pattern = r'''
        (@model|@inv|@gate)     |
        (⇒)                     |
        (::|:=)                 |
        ([A-Za-z_][A-Za-z0-9_]*) |
        (\d+\.?\d*)             |
        ([+\-*/=;,()\[\]])      |
        (\s+)                   |
        (.)
    '''
    
    for match in re.finditer(pattern, normalized, re.VERBOSE | re.DOTALL):
        groups = match.groups()
        token = None
        token_type = None
        
        # Find which group matched
        for i, g in enumerate(groups):
            if g is not None:
                token = g
                break
        
        if token is None or token.isspace():
            continue
        
        start = match.start()
        end = match.end()
        
        # Classify token
        if groups[0] is not None:
            if token == '@model':
                token_type = 'KW_MODEL'
            elif token == '@inv':
                token_type = 'KW_INV'
            elif token == '@gate':
                token_type = 'KW_GATE'
        elif groups[1] is not None:
            token_type = TOK_ARROW
        elif groups[2] is not None:
            if token == '::':
                token_type = TOK_COLON_COLON
            elif token == ':=':
                token_type = TOK_ASSIGN
        elif groups[3] is not None:
            if token in MODELS:
                token_type = 'MODEL'
            elif token in TYPES:
                token_type = 'TYPE'
            elif token in OPERATORS:
                token_type = 'OP'
            elif token == 'J':
                token_type = 'KW_J'
            else:
                token_type = TOK_IDENT
        elif groups[4] is not None:
            token_type = TOK_NUMBER
        elif groups[5] is not None:
            char = token
            type_map = {
                '+': TOK_PLUS, '-': TOK_MINUS, '*': TOK_STAR, '/': TOK_SLASH,
                '=': TOK_EQUAL, ';': TOK_SEMICOLON, ',': TOK_COMMA,
                '(': TOK_LPAREN, ')': TOK_RPAREN, '[': TOK_LBRACKET,
                ']': TOK_RBRACKET, ':': TOK_COLON
            }
            token_type = type_map.get(char, 'GLYPH')
        elif groups[7] is not None:
            token_type = 'GLYPH'
        
        if token_type:
            tokens.append({
                'type': token_type,
                'value': token,
                'start': start,
                'end': end
            })
    
    return tokens
