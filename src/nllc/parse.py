from dataclasses import dataclass
from typing import List
from .lex import Token, TokenKind, tokenize
from .ast import *
from .ast import BreakStmt

class ParseError(Exception):
    pass

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.current()
        self.pos += 1
        return tok

    def peek(self) -> Token:
        if self.pos + 1 < len(self.tokens):
            return self.tokens[self.pos + 1]
        return self.tokens[-1]  # EOF

    def expect(self, kind: TokenKind) -> Token:
        if self.current().kind != kind:
            raise ParseError(f"Expected {kind}, got {self.current().kind} at {self.current().span}")
        return self.advance()

    def at_end(self) -> bool:
        return self.current().kind == TokenKind.EOF

    def parse_program(self) -> Program:
        start = self.tokens[0].span.start
        statements = []
        while not self.at_end():
            statements.append(self.parse_statement())
        end = self.tokens[self.pos - 1].span.end
        return Program(Span(start, end), statements)

    def parse_statement(self) -> Statement:
        tok = self.current()
        if tok.kind == TokenKind.IMPORT:
            return self.parse_import()
        elif tok.kind == TokenKind.LET:
            return self.parse_let()
        elif tok.kind == TokenKind.MUT:
            return self.parse_mut()
        elif tok.kind == TokenKind.FN:
            return self.parse_fn()
        elif tok.kind == TokenKind.IF:
            return self.parse_if()
        elif tok.kind == TokenKind.WHILE:
            return self.parse_while()
        elif tok.kind == TokenKind.BREAK:
            return self.parse_break()
        elif tok.kind == TokenKind.RETURN:
            return self.parse_return()
        elif tok.kind == TokenKind.THREAD:
            return self.parse_thread()
        # NSC-M3L Phase 2: New statement types
        elif tok.kind == TokenKind.DIALECT:
            return self.parse_dialect()
        elif tok.kind == TokenKind.FIELD:
            return self.parse_field()
        elif tok.kind == TokenKind.TENSOR:
            return self.parse_tensor()
        elif tok.kind == TokenKind.METRIC:
            return self.parse_metric()
        elif tok.kind == TokenKind.INVARIANT:
            return self.parse_invariant()
        elif tok.kind == TokenKind.GAUGE:
            return self.parse_gauge()
        elif tok.kind == TokenKind.IDENT:
            start_pos = self.pos
            try:
                lvalue = self.parse_lvalue()
                if self.current().kind == TokenKind.ASSIGN:
                    self.expect(TokenKind.ASSIGN)
                    expr = self.parse_expr()
                    self.expect(TokenKind.SEMI)
                    end = self.current().span.end
                    return AssignStmt(Span(tok.span.start, end), lvalue, expr)
                else:
                    # backtrack
                    self.pos = start_pos
                    return self.parse_expr_stmt()
            except ParseError:
                self.pos = start_pos
                return self.parse_expr_stmt()
        else:
            return self.parse_expr_stmt()

    # NSC-M3L Phase 2: New parsing methods

    def parse_dialect(self) -> DialectStmt:
        """Parse dialect declaration: dialect ns | gr | ym;"""
        start = self.current().span.start
        self.expect(TokenKind.DIALECT)
        name_tok = self.expect(TokenKind.IDENT)  # ns, gr, or ym
        self.expect(TokenKind.SEMI)
        end = self.current().span.end
        return DialectStmt(Span(start, end), name_tok.value)

    def parse_type_annotation(self) -> TypeExpr:
        """Parse type annotation: Vector | Scalar | Tensor(symmetric) | Field[Vector]"""
        start = self.current().span.start
        
        # First type-like token is the base type
        if self.current().kind == TokenKind.IDENT:
            type_name = self.current().value
            self.advance()
            modifiers = []
        elif self.current().kind in [TokenKind.VECTOR, TokenKind.SCALAR, 
                                       TokenKind.TENSOR, TokenKind.FIELD]:
            type_name = self.current().value
            self.advance()
            modifiers = []
        else:
            raise ParseError(f"Expected type name, got {self.current().kind}")
        
        # Check for type modifiers after the base type
        while self.current().kind in [TokenKind.SYMMETRIC, TokenKind.ANTISYMMETRIC]:
            modifiers.append(self.current().value)
            self.advance()
        
        return TypeExpr(
            Span(start, self.pos - 1),
            name=type_name,
            modifiers=modifiers
        )

    def parse_field(self) -> FieldDecl:
        """Parse field declaration: field u: Vector;"""
        start = self.current().span.start
        self.expect(TokenKind.FIELD)
        name_tok = self.expect(TokenKind.IDENT)
        self.expect(TokenKind.COLON)
        field_type = self.parse_type_annotation()
        self.expect(TokenKind.SEMI)
        end = self.current().span.end
        return FieldDecl(Span(start, end), name_tok.value, field_type)

    def parse_tensor(self) -> TensorDecl:
        """Parse tensor declaration: tensor F: symmetric;"""
        start = self.current().span.start
        self.expect(TokenKind.TENSOR)
        name_tok = self.expect(TokenKind.IDENT)
        self.expect(TokenKind.COLON)
        tensor_type = self.parse_type_annotation()
        self.expect(TokenKind.SEMI)
        end = self.current().span.end
        return TensorDecl(Span(start, end), name_tok.value, tensor_type)

    def parse_metric(self) -> MetricDecl:
        """Parse metric declaration: metric g: Schwarzschild;"""
        start = self.current().span.start
        self.expect(TokenKind.METRIC)
        name_tok = self.expect(TokenKind.IDENT)
        self.expect(TokenKind.COLON)
        metric_type_tok = self.expect(TokenKind.IDENT)
        self.expect(TokenKind.SEMI)
        end = self.current().span.end
        return MetricDecl(Span(start, end), name_tok.value, metric_type_tok.value)

    def parse_invariant(self) -> InvariantStmt:
        """Parse invariant: invariant div_free [with EXPR] [require GATES];"""
        start = self.current().span.start
        self.expect(TokenKind.INVARIANT)
        name_tok = self.expect(TokenKind.IDENT)
        
        # Make 'with' clause optional
        constraint = None
        if self.current().kind == TokenKind.WITH:
            self.expect(TokenKind.WITH)
            constraint = self.parse_expr()
        
        # Make 'require' clause optional
        gates = []
        if self.current().kind == TokenKind.REQUIRE:
            self.expect(TokenKind.REQUIRE)
            while self.current().kind != TokenKind.SEMI:
                if self.current().kind == TokenKind.CONS:
                    gates.append('cons')
                    self.advance()
                elif self.current().kind == TokenKind.SEM:
                    gates.append('sem')
                    self.advance()
                elif self.current().kind == TokenKind.PHY:
                    gates.append('phy')
                    self.advance()
                else:
                    raise ParseError(f"Expected cons, sem, or phy, got {self.current().kind}")
                
                if self.current().kind == TokenKind.COMMA:
                    self.advance()
        
        self.expect(TokenKind.SEMI)
        end = self.current().span.end
        
        # Create a dummy constraint if not provided
        if constraint is None:
            zero = IntLit(Span(start, start), 0)
            constraint = zero
        
        return InvariantStmt(Span(start, end), name_tok.value, constraint, gates)

    def parse_gauge(self) -> GaugeStmt:
        """Parse gauge: gauge coulomb with EXPR require GATES;"""
        start = self.current().span.start
        self.expect(TokenKind.GAUGE)
        name_tok = self.expect(TokenKind.IDENT)
        self.expect(TokenKind.WITH)
        condition = self.parse_expr()
        self.expect(TokenKind.REQUIRE)
        
        gates = []
        while self.current().kind != TokenKind.SEMI:
            if self.current().kind == TokenKind.CONS:
                gates.append('cons')
            elif self.current().kind == TokenKind.PHY:
                gates.append('phy')
            else:
                raise ParseError(f"Expected cons or phy, got {self.current().kind}")
            self.advance()
            if self.current().kind == TokenKind.COMMA:
                self.advance()
        
        self.expect(TokenKind.SEMI)
        end = self.current().span.end
        
        return GaugeStmt(Span(start, end), name_tok.value, condition, gates)

    # Existing parsing methods (unchanged)

    def parse_import(self) -> ImportStmt:
        start = self.current().span.start
        self.expect(TokenKind.IMPORT)
        name_tok = self.expect(TokenKind.IDENT)
        self.expect(TokenKind.SEMI)
        end = self.current().span.end
        return ImportStmt(Span(start, end), name_tok.value)

    def parse_let(self) -> LetStmt:
        start = self.current().span.start
        self.expect(TokenKind.LET)
        var_tok = self.expect(TokenKind.IDENT)
        self.expect(TokenKind.ASSIGN)
        expr = self.parse_expr()
        self.expect(TokenKind.SEMI)
        end = self.current().span.end
        return LetStmt(Span(start, end), var_tok.value, expr)

    def parse_lvalue(self) -> Expr:
        tok = self.current()
        if tok.kind == TokenKind.IDENT:
            name = tok.value
            self.advance()
            expr = Var(tok.span, name)
            while self.current().kind == TokenKind.LBRACKET:
                self.expect(TokenKind.LBRACKET)
                index = self.parse_expr()
                self.expect(TokenKind.RBRACKET)
                expr = Index(Span(expr.span.start, self.current().span.end), expr, index)
            return expr
        else:
            raise ParseError("Expected lvalue")

    def parse_mut(self) -> MutStmt:
        start = self.current().span.start
        self.expect(TokenKind.MUT)
        var_tok = self.expect(TokenKind.IDENT)
        self.expect(TokenKind.ASSIGN)
        expr = self.parse_expr()
        self.expect(TokenKind.SEMI)
        end = self.current().span.end
        return MutStmt(Span(start, end), var_tok.value, expr)

    def parse_assign(self) -> AssignStmt:
        start = self.current().span.start
        lvalue = self.parse_lvalue()
        self.expect(TokenKind.ASSIGN)
        expr = self.parse_expr()
        self.expect(TokenKind.SEMI)
        end = self.current().span.end
        return AssignStmt(Span(start, end), lvalue, expr)

    def parse_fn(self) -> FnDecl:
        start = self.current().span.start
        self.expect(TokenKind.FN)
        name_tok = self.expect(TokenKind.IDENT)
        self.expect(TokenKind.LPAREN)
        params = []
        while self.current().kind != TokenKind.RPAREN:
            param_tok = self.expect(TokenKind.IDENT)
            params.append(param_tok.value)
            if self.current().kind == TokenKind.COLON:
                self.expect(TokenKind.COLON)
                self.expect(TokenKind.IDENT)
            if self.current().kind == TokenKind.COMMA:
                self.advance()
        self.expect(TokenKind.RPAREN)
        if self.current().kind == TokenKind.MINUS:
            if self.peek().kind == TokenKind.GT:
                self.advance()
                self.expect(TokenKind.GT)
                self.expect(TokenKind.IDENT)
        self.expect(TokenKind.LBRACE)
        body = []
        while self.current().kind != TokenKind.RBRACE:
            body.append(self.parse_statement())
        self.expect(TokenKind.RBRACE)
        end = self.current().span.end
        return FnDecl(Span(start, end), name_tok.value, params, body)

    def parse_if(self) -> IfStmt:
        start = self.current().span.start
        self.expect(TokenKind.IF)
        cond = self.parse_expr()
        self.expect(TokenKind.LBRACE)
        body = []
        while self.current().kind != TokenKind.RBRACE:
            body.append(self.parse_statement())
        self.expect(TokenKind.RBRACE)
        else_body = None
        if self.current().kind == TokenKind.ELSE:
            self.expect(TokenKind.ELSE)
            self.expect(TokenKind.LBRACE)
            else_body = []
            while self.current().kind != TokenKind.RBRACE:
                else_body.append(self.parse_statement())
            self.expect(TokenKind.RBRACE)
        end = self.current().span.end
        return IfStmt(Span(start, end), cond, body, else_body)

    def parse_while(self) -> WhileStmt:
        start = self.current().span.start
        self.expect(TokenKind.WHILE)
        cond = self.parse_expr()
        self.expect(TokenKind.LBRACE)
        body = []
        while self.current().kind != TokenKind.RBRACE:
            body.append(self.parse_statement())
        self.expect(TokenKind.RBRACE)
        end = self.current().span.end
        return WhileStmt(Span(start, end), cond, body)

    def parse_return(self) -> ReturnStmt:
        start = self.current().span.start
        self.expect(TokenKind.RETURN)
        expr = None
        if self.current().kind != TokenKind.SEMI:
            expr = self.parse_expr()
        self.expect(TokenKind.SEMI)
        end = self.current().span.end
        return ReturnStmt(Span(start, end), expr)

    def parse_break(self) -> BreakStmt:
        start = self.current().span.start
        self.expect(TokenKind.BREAK)
        self.expect(TokenKind.SEMI)
        end = self.current().span.end
        return BreakStmt(Span(start, end))

    def parse_expr_stmt(self) -> ExprStmt:
        start = self.current().span.start
        expr = self.parse_expr()
        self.expect(TokenKind.SEMI)
        end = self.current().span.end
        return ExprStmt(Span(start, end), expr)

    def parse_thread(self) -> ThreadBlock:
        start = self.current().span.start
        self.expect(TokenKind.THREAD)
        domain_tok = self.expect(TokenKind.IDENT)
        self.expect(TokenKind.DOT)
        scale_tok = self.expect(TokenKind.IDENT)
        self.expect(TokenKind.DOT)
        phase_tok = self.expect(TokenKind.IDENT)
        self.expect(TokenKind.LBRACE)
        body = []
        while self.current().kind != TokenKind.RBRACE:
            body.append(self.parse_statement())
        self.expect(TokenKind.RBRACE)
        self.expect(TokenKind.WITH)
        self.expect(TokenKind.REQUIRE)
        require = []
        while self.current().kind != TokenKind.SEMI:
            if self.current().kind == TokenKind.AUDIT:
                require.append('audit')
                self.advance()
            elif self.current().kind == TokenKind.ROLLBACK:
                require.append('rollback')
                self.advance()
            else:
                raise ParseError(f"Expected audit or rollback, got {self.current().kind}")
            if self.current().kind == TokenKind.COMMA:
                self.advance()
        self.expect(TokenKind.SEMI)
        end = self.current().span.end
        return ThreadBlock(Span(start, end), domain_tok.value, scale_tok.value, phase_tok.value, body, require)

    def parse_expr(self) -> Expr:
        return self.parse_unary()

    def parse_unary(self) -> Expr:
        if self.current().kind == TokenKind.MINUS:
            start = self.current().span.start
            self.advance()
            expr = self.parse_unary()
            end = expr.span.end
            zero = IntLit(Span(start, start), 0)
            return BinOp(Span(start, end), zero, '-', expr)
        else:
            return self.parse_binop()

    def parse_binop(self) -> Expr:
        left = self.parse_primary()
        while self.current().kind in [TokenKind.PLUS, TokenKind.MINUS, TokenKind.MUL, TokenKind.DIV, TokenKind.EQ, TokenKind.NE, TokenKind.LT, TokenKind.GT, TokenKind.LE, TokenKind.GE, TokenKind.AND]:
            op_tok = self.advance()
            right = self.parse_primary()
            left = BinOp(Span(left.span.start, right.span.end), left, op_tok.value, right)
        return left

    def parse_primary(self) -> Expr:
        expr = self.parse_atom()
        while self.current().kind == TokenKind.LBRACKET:
            self.expect(TokenKind.LBRACKET)
            index = self.parse_expr()
            self.expect(TokenKind.RBRACKET)
            expr = Index(Span(expr.span.start, self.current().span.end), expr, index)
        return expr

    def parse_atom(self) -> Expr:
        tok = self.current()
        
        # NSC-M3L Phase 2: Physics operators
        if tok.kind == TokenKind.DIVERGENCE:
            start = tok.span.start
            self.expect(TokenKind.DIVERGENCE)
            self.expect(TokenKind.LPAREN)
            arg = self.parse_expr()
            self.expect(TokenKind.RPAREN)
            return Divergence(Span(start, self.current().span.end), arg)
        
        elif tok.kind == TokenKind.CURL:
            start = tok.span.start
            self.expect(TokenKind.CURL)
            self.expect(TokenKind.LPAREN)
            arg = self.parse_expr()
            self.expect(TokenKind.RPAREN)
            return Curl(Span(start, self.current().span.end), arg)
        
        elif tok.kind == TokenKind.LAPLACIAN:
            start = tok.span.start
            self.expect(TokenKind.LAPLACIAN)
            self.expect(TokenKind.LPAREN)
            arg = self.parse_expr()
            self.expect(TokenKind.RPAREN)
            return Laplacian(Span(start, self.current().span.end), arg)
        
        elif tok.kind == TokenKind.GRAD:
            start = tok.span.start
            self.expect(TokenKind.GRAD)
            self.expect(TokenKind.LPAREN)
            arg = self.parse_expr()
            self.expect(TokenKind.RPAREN)
            return Gradient(Span(start, self.current().span.end), arg)
        
        elif tok.kind == TokenKind.TRACE:
            start = tok.span.start
            self.expect(TokenKind.TRACE)
            self.expect(TokenKind.LPAREN)
            arg = self.parse_expr()
            self.expect(TokenKind.RPAREN)
            return Trace(Span(start, self.current().span.end), arg)
        
        elif tok.kind == TokenKind.DET:
            start = tok.span.start
            self.expect(TokenKind.DET)
            self.expect(TokenKind.LPAREN)
            arg = self.parse_expr()
            self.expect(TokenKind.RPAREN)
            return Determinant(Span(start, self.current().span.end), arg)
        
        elif tok.kind == TokenKind.CONTRACT:
            start = tok.span.start
            self.expect(TokenKind.CONTRACT)
            self.expect(TokenKind.LPAREN)
            arg = self.parse_expr()
            # Parse optional index list
            indices = []
            if self.current().kind == TokenKind.COMMA:
                self.advance()
                while self.current().kind != TokenKind.RPAREN:
                    if self.current().kind == TokenKind.IDENT:
                        indices.append(self.current().value)
                        self.advance()
                    if self.current().kind == TokenKind.COMMA:
                        self.advance()
            self.expect(TokenKind.RPAREN)
            return Contraction(Span(start, self.current().span.end), arg, indices)
        
        elif tok.kind == TokenKind.CALL:
            start = tok.span.start
            self.expect(TokenKind.CALL)
            func_tok = self.expect(TokenKind.IDENT)
            func_name = func_tok.value
            self.expect(TokenKind.LPAREN)
            args = []
            while self.current().kind != TokenKind.RPAREN:
                args.append(self.parse_expr())
                if self.current().kind == TokenKind.COMMA:
                    self.advance()
            self.expect(TokenKind.RPAREN)
            end = self.current().span.end
            return Call(Span(start, end), func_name, args)
        
        elif tok.kind == TokenKind.INT:
            self.advance()
            return IntLit(tok.span, int(tok.value))
        
        elif tok.kind == TokenKind.FLOAT:
            self.advance()
            return FloatLit(tok.span, float(tok.value))
        
        elif tok.kind == TokenKind.TRUE:
            self.advance()
            return BoolLit(tok.span, True)
        
        elif tok.kind == TokenKind.FALSE:
            self.advance()
            return BoolLit(tok.span, False)
        
        elif tok.kind == TokenKind.STRING:
            self.advance()
            value = tok.value[1:-1]
            return StrLit(tok.span, value)
        
        elif tok.kind == TokenKind.IDENT:
            name = tok.value
            self.advance()
            if self.current().kind == TokenKind.LPAREN:
                self.expect(TokenKind.LPAREN)
                args = []
                while self.current().kind != TokenKind.RPAREN:
                    args.append(self.parse_expr())
                    if self.current().kind == TokenKind.COMMA:
                        self.advance()
                self.expect(TokenKind.RPAREN)
                return Call(Span(tok.span.start, self.current().span.end), name, args)
            else:
                return Var(tok.span, name)
        
        elif tok.kind == TokenKind.LBRACKET:
            start = tok.span.start
            self.expect(TokenKind.LBRACKET)
            elements = []
            while self.current().kind != TokenKind.RBRACKET:
                elements.append(self.parse_expr())
                if self.current().kind == TokenKind.COMMA:
                    self.advance()
            self.expect(TokenKind.RBRACKET)
            end = self.current().span.end
            return ArrayLit(Span(start, end), elements)
        
        elif tok.kind == TokenKind.LBRACE:
            return self.parse_object()
        
        elif tok.kind == TokenKind.LPAREN:
            self.expect(TokenKind.LPAREN)
            expr = self.parse_expr()
            self.expect(TokenKind.RPAREN)
            return expr
        
        elif tok.kind == TokenKind.IF:
            start = tok.span.start
            self.expect(TokenKind.IF)
            cond = self.parse_expr()
            self.expect(TokenKind.LBRACE)
            body = self.parse_expr()
            self.expect(TokenKind.RBRACE)
            else_body = None
            if self.current().kind == TokenKind.ELSE:
                self.expect(TokenKind.ELSE)
                self.expect(TokenKind.LBRACE)
                else_body = self.parse_expr()
                self.expect(TokenKind.RBRACE)
            end = self.current().span.end
            return IfExpr(Span(start, end), cond, body, else_body)
        
        else:
            raise ParseError(f"Unexpected token in atom: {tok.kind}")

    def parse_object(self) -> Expr:
        start = self.current().span.start
        self.expect(TokenKind.LBRACE)
        fields = {}
        while self.current().kind != TokenKind.RBRACE:
            if self.current().kind in (TokenKind.STRING, TokenKind.IDENT):
                key_tok = self.advance()
                key = key_tok.value.strip('"') if key_tok.kind == TokenKind.STRING else key_tok.value
            else:
                raise ParseError(f"Expected string or ident for key, got {self.current().kind}")
            self.expect(TokenKind.COLON)
            value = self.parse_expr()
            fields[key] = value
            if self.current().kind == TokenKind.COMMA:
                self.advance()
        self.expect(TokenKind.RBRACE)
        end = self.current().span.end
        return ObjectLit(Span(start, end), fields)

def parse(source: str) -> Program:
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse_program()
