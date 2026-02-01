"""
NSC-M3L Parser

Implements the full EBNF grammar from specifications/nsc_m3l_v1.md:
"""

from typing import Union, List, Optional, Set, Dict
from .ast import (
    Node, Span, Program, Statement,
    Atom, Group, OpCall, BinaryOp, Expr,
    Type, ScalarType, VectorType, TensorType, FieldType, FormType, OperatorType,
    ManifoldType, MetricType, LieAlgebraType, ConnectionType,
    Meta, Decl, Equation, Binding, Functional, Constraint, Predicate,
    Directive, DirectiveType, ModelSelector, InvariantList, GateSpec, TargetList,
    Model, SmoothnessClass, SemanticType
)


# Error constants
E_PARSE_EOF = 6
E_PARSE_UNEXPECTED_TOKEN = 7
E_PARSE_UNCLOSED_GROUP = 8
E_PARSE_TRAILING_INPUT = 9
E_PARSE_EMPTY_SENTENCE = 10
E_PARSE_MALFORMED_DECL = 11
E_PARSE_MALFORMED_EQUATION = 12
E_PARSE_MALFORMED_DIRECTIVE = 13


class ParseError(Exception):
    """Parser error with location information."""
    
    def __init__(self, message: str, pos: int = 0):
        self.message = message
        self.pos = pos
        super().__init__(f"Parse error at position {pos}: {message}")


class Parser:
    """
    NSC-M3L Parser implementing the full grammar.
    """
    
    def __init__(self, tokens: List[tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0
    
    def at_end(self) -> bool:
        return self.pos >= len(self.tokens)
    
    def current(self) -> tuple[str, str]:
        """Return current token as (type, value)."""
        if self.at_end():
            raise ParseError("Unexpected end of input", self.pos)
        return self.tokens[self.pos]
    
    def current_type(self) -> str:
        """Return current token type."""
        return self.current()[0]
    
    def current_value(self) -> str:
        """Return current token value."""
        return self.current()[1]
    
    def advance(self) -> tuple[str, str]:
        """Consume and return current token."""
        token = self.current()
        self.pos += 1
        return token
    
    def expect(self, expected_type: str, expected_value: Optional[str] = None) -> tuple[str, str]:
        """Consume token if it matches expected type and value."""
        token_type, token_value = self.current()
        if token_type != expected_type:
            raise ParseError(f"Expected {expected_type}, got {token_type}", self.pos)
        if expected_value is not None and token_value != expected_value:
            raise ParseError(f"Expected '{expected_value}', got '{token_value}'", self.pos)
        return self.advance()
    
    def match(self, expected_type: str, expected_value: Optional[str] = None) -> bool:
        """Check if current token matches without consuming."""
        if self.at_end():
            return False
        token_type, token_value = self.current()
        if token_type != expected_type:
            return False
        if expected_value is not None and token_value != expected_value:
            return False
        return True
    
    def parse_program(self) -> Program:
        """Parse complete NSC program."""
        start = self.pos
        statements = []
        
        while not self.at_end():
            try:
                statements.append(self.parse_statement())
            except ParseError as e:
                # Skip to next statement on error
                self.pos += 1
                continue
        
        end = self.pos
        return Program(start=start, end=end, statements=statements)
    
    def parse_statement(self) -> Statement:
        """Parse a statement (Decl, Equation, Functional, Constraint, or Directive)."""
        start = self.pos
        
        if self.at_end():
            raise ParseError("Empty statement", start)
        
        # Check for directives first (they start with @ or ⇒)
        token_type, token_value = self.current()
        
        if token_type == 'KW_MODEL':
            return self.parse_directive(start)
        elif token_type == 'KW_INV':
            return self.parse_directive(start)
        elif token_type == 'KW_GATE':
            return self.parse_directive(start)
        elif token_type == 'TOK_ARROW':
            return self.parse_directive(start)
        elif token_type == 'KW_J':
            return self.parse_functional(start)
        elif token_type == 'TOK_IDENT' and token_value == 'C':
            # Potential constraint, but check if followed by '('
            next_token = self.peek()
            if next_token and next_token[0] == 'TOK_LPAREN':
                return self.parse_constraint(start)
        
        # Check for declaration (look ahead for :: without consuming)
        if token_type == 'TOK_IDENT':
            ident = token_value
            next_token = self.peek()
            if next_token and next_token[0] == 'TOK_COLON_COLON':
                self.advance()  # consume identifier
                return self.parse_decl(start, ident)
        
        # Otherwise, parse as equation
        return self.parse_equation(start)
    
    def parse_decl(self, start: int, ident: Optional[str] = None) -> Decl:
        """Parse declaration: Ident :: Type [ ":" Meta ] ";" """
        if ident is None:
            ident = self.expect('TOK_IDENT')[1]
        
        self.expect('TOK_COLON_COLON')
        decl_type = self.parse_type()
        
        # Optional metadata
        meta = None
        if self.match('TOK_COLON'):
            self.advance()
            meta = self.parse_meta()
        
        self.expect('TOK_SEMICOLON')
        end = self.pos
        
        return Decl(start=start, end=end, ident=ident, decl_type=decl_type, meta=meta)
    
    def parse_type(self) -> Type:
        """Parse type: Scalar | Vector | TensorType | FieldType | FormType | OperatorType | ManifoldType | MetricType | LieAlgebraType"""
        start = self.pos
        token_type = self.current_type()
        
        if token_type == 'TYPE':
            type_name = self.current_value()
            self.advance()
            
            if type_name == 'Scalar':
                return ScalarType(start=start, end=self.pos)
            elif type_name == 'Vector':
                # Check for optional dimension: Vector[n]
                if self.match('TOK_LBRACKET'):
                    self.advance()
                    dim_token = self.expect('TOK_NUMBER')
                    dim = int(dim_token[1])
                    self.expect('TOK_RBRACKET')
                    return VectorType(start=start, end=self.pos, dim=dim)
                return VectorType(start=start, end=self.pos)
            elif type_name == 'Tensor':
                # Tensor(k, l) syntax
                self.expect('TOK_LPAREN')
                k = int(self.expect('TOK_NUMBER')[1])
                self.expect('TOK_COMMA')
                l = int(self.expect('TOK_NUMBER')[1])
                self.expect('TOK_RPAREN')
                return TensorType(start=start, end=self.pos, k=k, l=l)
            elif type_name == 'Field':
                # Field[Type] syntax
                self.expect('TOK_LBRACKET')
                value_type = self.parse_type()
                self.expect('TOK_RBRACKET')
                return FieldType(start=start, end=self.pos, value_type=value_type)
            elif type_name == 'Form':
                # Form[p] syntax
                self.expect('TOK_LBRACKET')
                p = int(self.expect('TOK_NUMBER')[1])
                self.expect('TOK_RBRACKET')
                return FormType(start=start, end=self.pos, p=p)
            elif type_name == 'Operator':
                # Operator(Domain -> Codomain) syntax
                self.expect('TOK_LPAREN')
                domain = self.parse_type()
                self.expect('TOK_ARROW')
                codomain = self.parse_type()
                self.expect('TOK_RPAREN')
                return OperatorType(start=start, end=self.pos, domain=domain, codomain=codomain)
            elif type_name == 'Manifold':
                # Manifold(dim, signature) syntax
                self.expect('TOK_LPAREN')
                dim = int(self.expect('TOK_NUMBER')[1])
                signature = None
                if self.match('TOK_COMMA'):
                    self.advance()
                    # Parse signature identifier
                    sig_token = self.expect('TOK_IDENT')
                    signature = sig_token[1]
                self.expect('TOK_RPAREN')
                return ManifoldType(start=start, end=self.pos, dim=dim, signature=signature)
            elif type_name == 'Metric':
                # Metric(dim, signature) syntax (optional params)
                metric_type = MetricType(start=start, end=self.pos)
                if self.match('TOK_LPAREN'):
                    self.advance()
                    if self.match('TOK_NUMBER'):
                        dim = int(self.expect('TOK_NUMBER')[1])
                        metric_type.dim = dim
                        if self.match('TOK_COMMA'):
                            self.advance()
                            sig_token = self.expect('TOK_IDENT')
                            metric_type.signature = sig_token[1]
                    elif self.match('TOK_IDENT'):
                        sig_token = self.expect('TOK_IDENT')
                        metric_type.signature = sig_token[1]
                    self.expect('TOK_RPAREN')
                return metric_type
            elif type_name == 'LieAlgebra':
                # LieAlgebra(name) syntax
                self.expect('TOK_LPAREN')
                name_token = self.expect('TOK_IDENT')
                self.expect('TOK_RPAREN')
                return LieAlgebraType(start=start, end=self.pos, name=name_token[1])
            elif type_name == 'Connection':
                # Connection syntax (optional flags)
                conn_type = ConnectionType(start=start, end=self.pos)
                if self.match('TOK_LPAREN'):
                    self.advance()
                    # Parse options
                    while not self.match('TOK_RPAREN'):
                        if self.match('TOK_IDENT'):
                            key = self.expect('TOK_IDENT')[1]
                            if key == 'metric_compatible':
                                conn_type.metric_compatible = True
                            elif key == 'torsion_free':
                                conn_type.torsion_free = True
                        if self.match('TOK_COMMA'):
                            self.advance()
                    self.expect('TOK_RPAREN')
                return conn_type
        
        raise ParseError(f"Expected type, got {token_type}", start)
    
    def parse_meta(self) -> Meta:
        """Parse metadata: { Key = Value }"""
        start = self.pos
        pairs: Dict[str, str] = {}
        
        while self.match('TOK_IDENT'):
            key = self.expect('TOK_IDENT')[1]
            self.expect('TOK_EQUAL')
            value = self.expect('TOK_IDENT')  # Could be other types too
            pairs[key] = value[1]
        
        return Meta(start=self.pos, end=self.pos, pairs=pairs)
    
    def parse_equation(self, start: int) -> Equation:
        """Parse equation: Expr = Expr [ ":" Meta ] ";" """
        lhs = self.parse_expr()
        
        self.expect('TOK_EQUAL')
        rhs = self.parse_expr()
        
        # Optional metadata
        meta = None
        if self.match('TOK_COLON'):
            self.advance()
            meta = self.parse_meta()
        
        self.expect('TOK_SEMICOLON')
        end = self.pos
        
        return Equation(start=start, end=end, lhs=lhs, rhs=rhs, meta=meta)
    
    def parse_functional(self, start: int) -> Functional:
        """Parse functional: J "(" Bindings ")" ":=" Expr [ ":" Meta ] ";" """
        self.expect('KW_J')
        self.expect('TOK_LPAREN')
        
        bindings = []
        while not self.match('TOK_RPAREN'):
            if bindings:
                self.expect('TOK_COMMA')
            ident = self.expect('TOK_IDENT')[1]
            binding_type = None
            if self.match('TOK_COLON_COLON'):
                self.advance()
                binding_type = self.parse_type()
            bindings.append(Binding(start=self.pos, end=self.pos, ident=ident, type=binding_type))
        
        self.expect('TOK_RPAREN')
        self.expect('TOK_ASSIGN')
        
        expr = self.parse_expr()
        
        # Optional metadata
        meta = None
        if self.match('TOK_COLON'):
            self.advance()
            meta = self.parse_meta()
        
        self.expect('TOK_SEMICOLON')
        end = self.pos
        
        return Functional(start=start, end=end, bindings=bindings, expr=expr, meta=meta)
    
    def parse_constraint(self, start: int) -> Constraint:
        """Parse constraint: C "(" Ident ")" ":=" Predicate [ ":" Meta ] ";" """
        self.expect('TOK_IDENT')  # C
        self.expect('TOK_LPAREN')
        ident = self.expect('TOK_IDENT')[1]
        self.expect('TOK_RPAREN')
        self.expect('TOK_ASSIGN')
        
        # Parse predicate (simple expression for now)
        predicate = self.parse_expr()
        
        # Optional metadata
        meta = None
        if self.match('TOK_COLON'):
            self.advance()
            meta = self.parse_meta()
        
        self.expect('TOK_SEMICOLON')
        end = self.pos
        
        return Constraint(start=start, end=end, ident=ident, predicate=predicate, meta=meta)
    
    def parse_directive(self, start: int) -> Directive:
        """Parse directive: @model(...) | @inv(...) | @gate(...) | ⇒ TargetList";" """
        token_type = self.current_type()
        
        if token_type == 'KW_MODEL':
            self.advance()
            self.expect('TOK_LPAREN')
            models = self.parse_model_list()
            self.expect('TOK_RPAREN')
            self.expect('TOK_SEMICOLON')
            end = self.pos
            return Directive(start=start, end=end, directive_type=DirectiveType.MODEL, 
                           model_selector=ModelSelector(start=start, end=end, models=models))
        
        elif token_type == 'KW_INV':
            self.advance()
            self.expect('TOK_LPAREN')
            invariants = self.parse_invariant_list()
            self.expect('TOK_RPAREN')
            self.expect('TOK_SEMICOLON')
            end = self.pos
            return Directive(start=start, end=end, directive_type=DirectiveType.INV,
                           invariant_list=InvariantList(start=start, end=end, invariants=invariants))
        
        elif token_type == 'KW_GATE':
            self.advance()
            self.expect('TOK_LPAREN')
            config = self.parse_gate_spec()
            self.expect('TOK_RPAREN')
            self.expect('TOK_SEMICOLON')
            end = self.pos
            return Directive(start=start, end=end, directive_type=DirectiveType.GATE,
                           gate_spec=GateSpec(start=start, end=end, config=config))
        
        elif token_type == 'TOK_ARROW':
            self.advance()
            targets = self.parse_target_list()
            self.expect('TOK_SEMICOLON')
            end = self.pos
            return Directive(start=start, end=end, directive_type=DirectiveType.COMPILE,
                           target_list=TargetList(start=start, end=end, targets=targets))
        
        raise ParseError(f"Unknown directive type: {token_type}", start)
    
    def parse_model_list(self) -> Set[Model]:
        """Parse model list: Model { "," Model }"""
        models = set()
        
        while True:
            token_type, token_value = self.current()
            if token_type == 'MODEL' and token_value in {'ALG', 'CALC', 'GEO', 'DISC', 'LEDGER', 'EXEC'}:
                models.add(Model(token_value))
                self.advance()
                if self.match('TOK_COMMA'):
                    self.advance()
                    continue
            break
        
        return models
    
    def parse_invariant_list(self) -> List[str]:
        """Parse invariant list for @inv directive."""
        invariants = []
        
        while not self.match('TOK_RPAREN'):
            if invariants:
                self.expect('TOK_COMMA')
            # Parse invariant reference (e.g., N:INV.gr.hamiltonian_constraint)
            if self.match('TOK_IDENT'):
                inv = self.expect('TOK_IDENT')[1]
                invariants.append(inv)
            else:
                raise ParseError("Expected invariant identifier", self.pos)
        
        return invariants
    
    def parse_gate_spec(self) -> Dict[str, str]:
        """Parse gate specification for @gate directive."""
        config = {}
        
        while not self.match('TOK_RPAREN'):
            if config:
                self.expect('TOK_COMMA')
            key = self.expect('TOK_IDENT')[1]
            self.expect('TOK_EQUAL')
            # Value can be IDENT or NUMBER
            value_token = self.current()
            if value_token[0] in ('TOK_IDENT', 'TOK_NUMBER'):
                self.advance()
                value = value_token[1]
            else:
                raise ParseError(f"Expected value, got {value_token[0]}", self.pos)
            config[key] = value
        
        return config
    
    def parse_target_list(self) -> Set[Model]:
        """Parse target list for ⇒ directive."""
        # Skip TOK_LPAREN if present
        if self.match('TOK_LPAREN'):
            self.advance()
        
        models = set()
        
        while True:
            token_type, token_value = self.current()
            if token_type == 'MODEL' and token_value in {'ALG', 'CALC', 'GEO', 'DISC', 'LEDGER', 'EXEC'}:
                models.add(Model(token_value))
                self.advance()
                if self.match('TOK_COMMA'):
                    self.advance()
                    continue
            break
        
        # Skip TOK_RPAREN if present
        if self.match('TOK_RPAREN'):
            self.advance()
        
        return models
    
    # === Expression Parsing ===
    
    def parse_expr(self) -> Expr:
        """Parse expression: Term { ("+"|"-") Term }"""
        start = self.pos
        left = self.parse_term()
        
        while self.match('TOK_PLUS') or self.match('TOK_MINUS'):
            op = self.advance()[1]
            right = self.parse_term()
            left = BinaryOp(start=self.pos, end=self.pos, op=op, left=left, right=right)
            start = self.pos  # Reset start for chained ops
        
        return left
    
    def parse_term(self) -> Expr:
        """Parse term: Factor { ("*"|"/") Factor }"""
        start = self.pos
        left = self.parse_factor()
        
        while self.match('TOK_STAR') or self.match('TOK_SLASH'):
            op = self.advance()[1]
            right = self.parse_factor()
            left = BinaryOp(start=self.pos, end=self.pos, op=op, left=left, right=right)
            start = self.pos
        
        return left
    
    def parse_factor(self) -> Expr:
        """Parse factor: Atom | Op "(" Expr ")" | "(" Expr ")" """
        start = self.pos
        token_type = self.current_type()
        
        if token_type == 'OP':
            # Operator call: Op(Expr)
            op = self.advance()[1]
            self.expect('TOK_LPAREN')
            arg = self.parse_expr()
            self.expect('TOK_RPAREN')
            return OpCall(start=start, end=self.pos, op=op, arg=arg)
        
        elif token_type == 'TOK_LPAREN':
            # Group: (Expr)
            self.advance()
            inner = self.parse_expr()
            self.expect('TOK_RPAREN')
            return Group(start=start, end=self.pos, delim='()', inner=inner)
        
        elif token_type == 'TOK_LBRACKET':
            # Bracket group: [Expr]
            self.advance()
            inner = self.parse_expr()
            self.expect('TOK_RBRACKET')
            return Group(start=start, end=self.pos, delim='[]', inner=inner)
        
        else:
            # Atom: Ident | Number | Tensor | FieldAccess
            return self.parse_atom()
    
    def parse_atom(self) -> Atom:
        """Parse atom: Ident | Number | Tensor | FieldAccess"""
        start = self.pos
        token_type, token_value = self.current()
        
        if token_type in ('TOK_IDENT', 'TOK_NUMBER', 'GLYPH'):
            self.advance()
            return Atom(start=start, end=self.pos, value=token_value)
        
        raise ParseError(f"Expected atom, got {token_type}", start)
    
    def peek(self) -> Optional[tuple[str, str]]:
        """Look ahead without consuming."""
        if self.pos + 1 >= len(self.tokens):
            return None
        return self.tokens[self.pos + 1]


# === Public API ===

def parse_program(tokens: List[tuple[str, str]]) -> Program:
    """
    Parse NSC program from tokens.
    
    Args:
        tokens: List of (token_type, token_value) tuples
        
    Returns:
        Program AST node
    """
    parser = Parser(tokens)
    program = parser.parse_program()
    if not parser.at_end():
        raise ParseError("Trailing input after program", parser.pos)
    return program


def parse_string(source: str) -> Program:
    """
    Parse NSC program from source string.
    
    Args:
        source: NSC source code
        
    Returns:
        Program AST node
    """
    from .lex import tokenize
    tokens = tokenize(source)
    return parse_program(tokens)
