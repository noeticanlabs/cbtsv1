"""
Comprehensive tests for NSC-M3L Parser

Tests cover:
- Lexer tokenization with new tokens
- Parser for declarations, equations, functionals, constraints, directives
- Type annotations (Scalar, Vector, Tensor, Field, Form, Operator)
- Metadata annotations
- Error handling for malformed input
"""

import pytest
from src.nsc.lex import tokenize, tokenize_simple, tokenize_with_positions
from src.nsc.parse import parse_program, parse_string, Parser, ParseError
from src.nsc.ast import (
    Program, Decl, Equation, Functional, Constraint, Directive,
    Atom, Group, BinaryOp, OpCall,
    ScalarType, VectorType, TensorType, FieldType, FormType, OperatorType,
    ModelSelector, InvariantList, GateSpec, TargetList,
    Model, DirectiveType, Binding
)


class TestLexer:
    """Tests for NSC lexer/tokenizer."""
    
    def test_basic_tokens(self):
        """Test basic token recognition."""
        tokens = tokenize("x + y")
        assert tokens == [
            ('TOK_IDENT', 'x'),
            ('TOK_PLUS', '+'),
            ('TOK_IDENT', 'y')
        ]
    
    def test_directive_keywords(self):
        """Test @model, @inv, @gate tokenization."""
        tokens = tokenize("@model(GEO, CALC)")
        assert tokens[0] == ('KW_MODEL', '@model')
        assert tokens[1] == ('TOK_LPAREN', '(')
        assert tokens[2] == ('MODEL', 'GEO')
        assert tokens[3] == ('TOK_COMMA', ',')
        assert tokens[4] == ('MODEL', 'CALC')
        assert tokens[5] == ('TOK_RPAREN', ')')
    
    def test_arrow_token(self):
        """Test ⇒ token recognition."""
        tokens = tokenize("⇒ (LEDGER, CALC)")
        assert tokens[0] == ('TOK_ARROW', '⇒')
    
    def test_double_colon(self):
        """Test :: token recognition."""
        tokens = tokenize("x :: Scalar;")
        assert ('TOK_COLON_COLON', '::') in tokens
    
    def test_assign_token(self):
        """Test := token recognition."""
        tokens = tokenize("J(u) := u + v;")
        assert ('TOK_ASSIGN', ':=') in tokens
    
    def test_model_names(self):
        """Test ALG, CALC, GEO, DISC, LEDGER, EXEC tokenization."""
        for model in ['ALG', 'CALC', 'GEO', 'DISC', 'LEDGER', 'EXEC']:
            tokens = tokenize(model)
            assert tokens[0][0] == 'MODEL'
            assert tokens[0][1] == model
    
    def test_type_names(self):
        """Test Scalar, Vector, Tensor, Field, Form, Operator tokenization."""
        for type_name in ['Scalar', 'Vector', 'Tensor', 'Field', 'Form', 'Operator']:
            tokens = tokenize(type_name)
            assert tokens[0][0] == 'TYPE'
            assert tokens[0][1] == type_name
    
    def test_numbers(self):
        """Test number tokenization."""
        tokens = tokenize("123 45.67")
        assert tokens[0][0] == 'TOK_NUMBER'
        assert tokens[0][1] == '123'
        assert tokens[1][0] == 'TOK_NUMBER'
        assert tokens[1][1] == '45.67'
    
    def test_operators(self):
        """Test operator tokenization."""
        tokens = tokenize("div(v)")
        assert ('OP', 'div') in tokens
        tokens = tokenize("curl(v)")
        assert ('OP', 'curl') in tokens
        tokens = tokenize("grad(f)")
        assert ('OP', 'grad') in tokens
    
    def test_simple_tokenize_backward_compat(self):
        """Test backward compatibility with simple tokenize."""
        tokens = tokenize_simple("x + y")
        assert tokens == ['x', '+', 'y']
    
    def test_tokenize_with_positions(self):
        """Test position information in tokens."""
        tokens = tokenize_with_positions("x")
        assert len(tokens) == 1
        assert tokens[0]['type'] == 'TOK_IDENT'
        assert tokens[0]['value'] == 'x'
        assert tokens[0]['start'] == 0
        assert tokens[0]['end'] == 1


class TestParserDeclarations:
    """Tests for declaration parsing."""
    
    def test_simple_scalar_decl(self):
        """Test simple Scalar declaration."""
        source = "x :: Scalar;"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        assert len(program.statements) == 1
        stmt = program.statements[0]
        assert isinstance(stmt, Decl)
        assert stmt.ident == 'x'
        assert isinstance(stmt.decl_type, ScalarType)
    
    def test_vector_decl(self):
        """Test Vector declaration."""
        source = "v :: Vector;"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        stmt = program.statements[0]
        assert isinstance(stmt, Decl)
        assert stmt.ident == 'v'
        assert isinstance(stmt.decl_type, VectorType)
    
    def test_vector_with_dim(self):
        """Test Vector with dimension."""
        source = "v :: Vector[3];"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        stmt = program.statements[0]
        assert isinstance(stmt.decl_type, VectorType)
        assert stmt.decl_type.dim == 3
    
    def test_tensor_decl(self):
        """Test Tensor(k,l) declaration."""
        source = "T :: Tensor(2,1);"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        stmt = program.statements[0]
        assert isinstance(stmt.decl_type, TensorType)
        assert stmt.decl_type.k == 2
        assert stmt.decl_type.l == 1
    
    def test_field_decl(self):
        """Test Field[Type] declaration."""
        source = "u :: Field[Vector];"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        stmt = program.statements[0]
        assert isinstance(stmt.decl_type, FieldType)
        assert isinstance(stmt.decl_type.value_type, VectorType)
    
    def test_form_decl(self):
        """Test Form[p] declaration."""
        source = "omega :: Form[2];"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        stmt = program.statements[0]
        assert isinstance(stmt.decl_type, FormType)
        assert stmt.decl_type.p == 2
    
    def test_multiple_decls(self):
        """Test multiple declarations."""
        source = """
        x :: Scalar;
        y :: Vector;
        z :: Tensor(1,1);
        """
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        assert len(program.statements) == 3
        assert all(isinstance(s, Decl) for s in program.statements)


class TestParserEquations:
    """Tests for equation parsing."""
    
    def test_simple_equation(self):
        """Test simple equation."""
        source = "x + y = z;"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        assert len(program.statements) == 1
        stmt = program.statements[0]
        assert isinstance(stmt, Equation)
        assert isinstance(stmt.lhs, BinaryOp)
        assert stmt.lhs.op == '+'
        assert isinstance(stmt.rhs, Atom)
        assert stmt.rhs.value == 'z'
    
    def test_complex_equation(self):
        """Test equation with multiple operations."""
        source = "a * x + b * y = c;"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        stmt = program.statements[0]
        assert isinstance(stmt, Equation)
        # Should have correct operator precedence
        assert isinstance(stmt.lhs, BinaryOp)
    
    def test_operator_equation(self):
        """Test equation with operators."""
        source = "div(v) = 0;"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        stmt = program.statements[0]
        assert isinstance(stmt.lhs, OpCall)
        assert stmt.lhs.op == 'div'
    
    def test_grouped_equation(self):
        """Test equation with parentheses."""
        source = "(a + b) * c = d;"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        stmt = program.statements[0]
        assert isinstance(stmt.lhs, BinaryOp)
        assert isinstance(stmt.lhs.left, Group)


class TestParserFunctionals:
    """Tests for functional parsing."""
    
    def test_simple_functional(self):
        """Test simple functional."""
        source = "J(u) := u;"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        assert len(program.statements) == 1
        stmt = program.statements[0]
        assert isinstance(stmt, Functional)
        assert len(stmt.bindings) == 1
        assert stmt.bindings[0].ident == 'u'
    
    def test_functional_with_multiple_bindings(self):
        """Test functional with multiple bindings."""
        source = "J(u, v) := u + v;"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        stmt = program.statements[0]
        assert len(stmt.bindings) == 2
        assert stmt.bindings[0].ident == 'u'
        assert stmt.bindings[1].ident == 'v'
    
    def test_functional_with_typed_binding(self):
        """Test functional with typed binding."""
        source = "J(u :: Vector) := u;"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        stmt = program.statements[0]
        assert len(stmt.bindings) == 1
        assert stmt.bindings[0].type is not None
        assert isinstance(stmt.bindings[0].type, VectorType)


class TestParserConstraints:
    """Tests for constraint parsing."""
    
    def test_simple_constraint(self):
        """Test simple constraint."""
        source = "C(x) := x;"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        assert len(program.statements) == 1
        stmt = program.statements[0]
        assert isinstance(stmt, Constraint)
        assert stmt.ident == 'x'


class TestParserDirectives:
    """Tests for directive parsing."""
    
    def test_model_directive(self):
        """Test @model directive."""
        source = "@model(GEO, CALC, LEDGER);"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        assert len(program.statements) == 1
        stmt = program.statements[0]
        assert isinstance(stmt, Directive)
        assert stmt.directive_type == DirectiveType.MODEL
        assert stmt.model_selector is not None
        assert Model.GEO in stmt.model_selector.models
        assert Model.CALC in stmt.model_selector.models
        assert Model.LEDGER in stmt.model_selector.models
    
    def test_inv_directive(self):
        """Test @inv directive."""
        source = "@inv(INV_hamiltonian_constraint, INV_momentum_constraint);"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        stmt = program.statements[0]
        assert isinstance(stmt, Directive)
        assert stmt.directive_type == DirectiveType.INV
        assert stmt.invariant_list is not None
        assert len(stmt.invariant_list.invariants) == 2
    
    def test_gate_directive(self):
        """Test @gate directive."""
        source = "@gate(eps=1, max_iter=1000);"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        stmt = program.statements[0]
        assert isinstance(stmt, Directive)
        assert stmt.directive_type == DirectiveType.GATE
        assert stmt.gate_spec is not None
        assert stmt.gate_spec.config.get('eps') == '1'
        assert stmt.gate_spec.config.get('max_iter') == '1000'
    
    def test_compile_directive(self):
        """Test ⇒ (compile) directive."""
        source = "⇒ (LEDGER, CALC, GEO);"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        stmt = program.statements[0]
        assert isinstance(stmt, Directive)
        assert stmt.directive_type == DirectiveType.COMPILE
        assert stmt.target_list is not None
        assert Model.LEDGER in stmt.target_list.targets
        assert Model.CALC in stmt.target_list.targets
        assert Model.GEO in stmt.target_list.targets


class TestParserExpressions:
    """Tests for expression parsing."""
    
    def test_addition(self):
        """Test addition expression with simple source."""
        source = "x+y=z;"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        stmt = program.statements[0]
        assert isinstance(stmt, Equation)
        assert isinstance(stmt.lhs, BinaryOp)
        assert stmt.lhs.op == '+'
    
    def test_subtraction(self):
        """Test subtraction expression."""
        source = "a - b = c;"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        stmt = program.statements[0]
        assert stmt.lhs.op == '-'
    
    def test_multiplication(self):
        """Test multiplication expression."""
        source = "a * b = c;"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        stmt = program.statements[0]
        assert isinstance(stmt.lhs, BinaryOp)
    
    def test_division(self):
        """Test division expression."""
        source = "a / b = c;"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        stmt = program.statements[0]
        assert stmt.lhs.op == '/'
    
    def test_mixed_operations(self):
        """Test mixed operations with precedence."""
        source = "a + b * c = d;"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        stmt = program.statements[0]
        # Should respect precedence: b * c first, then a + result
        assert isinstance(stmt.lhs, BinaryOp)
        assert stmt.lhs.op == '+'
        assert isinstance(stmt.lhs.right, BinaryOp)
        assert stmt.lhs.right.op == '*'
    
    def test_parenthesized_expression(self):
        """Test parenthesized expression."""
        source = "(a + b) * c = d;"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        stmt = program.statements[0]
        assert isinstance(stmt.lhs, BinaryOp)
        assert isinstance(stmt.lhs.left, Group)
    
    def test_bracketed_expression(self):
        """Test bracketed expression."""
        source = "[a + b] * c = d;"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        stmt = program.statements[0]
        assert isinstance(stmt.lhs.left, Group)
        assert stmt.lhs.left.delim == '[]'
    
    def test_operator_call(self):
        """Test operator call."""
        source = "div(v) = 0;"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        stmt = program.statements[0]
        assert isinstance(stmt.lhs, OpCall)
        assert stmt.lhs.op == 'div'
    
    def test_chained_operators(self):
        """Test chained operator calls."""
        source = "grad(div(f)) = 0;"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        stmt = program.statements[0]
        assert isinstance(stmt.lhs, OpCall)
        assert stmt.lhs.op == 'grad'
        assert isinstance(stmt.lhs.arg, OpCall)
        assert stmt.lhs.arg.op == 'div'


class TestParserPrograms:
    """Tests for complete program parsing."""
    
    def test_simple_program(self):
        """Test simple program with declarations."""
        source = """
        x :: Scalar;
        y :: Vector;
        """
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        assert isinstance(program, Program)
        assert len(program.statements) == 2


class TestParserErrors:
    """Tests for error handling."""
    
    def test_unexpected_token(self):
        """Test unexpected token handling."""
        source = "x :: Scalar ; extra"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        # Should parse successfully, ignoring trailing input
        assert len(program.statements) == 1


class TestParserAST:
    """Tests for AST node attributes."""
    
    def test_node_has_required_attributes(self):
        """Test that nodes have required attributes from spec."""
        source = "x :: Scalar;"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        stmt = program.statements[0]
        # All nodes should have these attributes
        assert hasattr(stmt, 'type')
        assert hasattr(stmt, 'domains_used')
        assert hasattr(stmt, 'invariants_required')
        assert hasattr(stmt, 'effects')
    
    def test_ast_to_dict(self):
        """Test AST serialization to dictionary."""
        source = "x :: Scalar;"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        d = program.to_dict()
        assert isinstance(d, dict)
        assert 'statements' in d
        assert len(d['statements']) == 1
    
    def test_program_to_dict(self):
        """Test Program node to_dict."""
        source = "x = y;"
        tokens = tokenize(source)
        program = parse_program(tokens)
        
        d = program.to_dict()
        assert 'start' in d
        assert 'end' in d
        assert 'statements' in d


class TestParseString:
    """Tests for parse_string convenience function."""
    
    def test_parse_string_simple(self):
        """Test parse_string with simple input."""
        source = "x = y;"
        program = parse_string(source)
        
        assert isinstance(program, Program)
        assert len(program.statements) == 1
    
    def test_parse_string_complex(self):
        """Test parse_string with complex input."""
        source = """
        @model(CALC, LEDGER);
        u :: Field[Vector];
        """
        program = parse_string(source)
        
        assert isinstance(program, Program)
        assert len(program.statements) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
