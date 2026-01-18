import re
import unicodedata

def tokenize(nsc: str) -> list[str]:
    normalized = unicodedata.normalize('NFC', nsc)
    if normalized != nsc:
        raise ValueError("Non-canonical Unicode")  # Adapted from nsc_diag.NSCError(nsc_diag.E_NONCANONICAL_UNICODE, ...)
    # Pattern to match operators and identifiers/numbers
    pattern = r'(\+|\-|\*|/|∂|∇²|∇|=|\(|\)|\[|\]|φ|⊕|↻|∆|◯|⊖|⇒|□|\w+|\d+\.?\d*)'
    tokens = re.findall(pattern, normalized)
    tokens = [t for t in tokens if t.strip()]  # remove empty
    if not tokens:
        raise ValueError("Empty input string")  # Adapted
    # Classify tokens
    for i, t in enumerate(tokens):
        if t == 'import':
            tokens[i] = 'KW_IMPORT'
        elif re.match(r'[A-Za-z_][A-Za-z0-9_]*', t):
            tokens[i] = 'IDENT'
        # else keep as is (operators, numbers, glyphs)
    return tokens