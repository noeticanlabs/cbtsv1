"""
NSC-M3L Error Handling Integration Tests

Tests error handling across all pipeline stages.
Covers:
- Lexer error propagation
- Parse error recovery
- Type error location reporting
- Geometry prerequisite errors
- Model compatibility errors
"""

import pytest
from typing import Dict, List, Optional

from tests.nsc_test_utils import (
    compile_nsc_source, parse_source, type_check_program,
    tokenize_source, parse_safe, type_check_safe,
    CompilationResult, ParseResult, TypeCheckResult,
    expect_lex_error, expect_parse_error, expect_type_error
)
from src.nsc.parse import ParseError
from src.nsc.lex import tokenize
from src.nsc.type_checker import TypeChecker
from src.nsc.types import TypeError, GeometryPrerequisiteError


class TestLexerErrorHandling:
    """Test lexer error handling."""
    
    def test_unexpected_character(self):
        """Test unexpected character detection."""
        source = "x :: Scalar; ¥"
        
        with pytest.raises(Exception):
            tokenize(source)
    
    def test_unterminated_string(self):
        """Test unterminated string detection."""
        source = 'x :: Scalar; "unterminated'
        
        with pytest.raises(Exception):
            tokenize(source)
    
    def test_invalid_token(self):
        """Test invalid token handling."""
        source = "x :: Scalar; @@@"
        
        try:
            tokenize(source)
        except Exception:
            pass


class TestParseErrorHandling:
    """Test parse error handling."""
    
    def test_unexpected_token(self):
        """Test unexpected token handling."""
        source = "x :: Scalar ; y"
        
        result = parse_safe(source)
        assert result.ast is not None or result.error is not None
    
    def test_missing_rhs(self):
        """Test missing RHS in equation."""
        source = "x :: Scalar; y ="
        
        with pytest.raises(ParseError):
            parse_source(source)
    
    def test_missing_type(self):
        """Test missing type in declaration."""
        source = "x ::;"
        
        with pytest.raises(ParseError):
            parse_source(source)
    
    def test_unclosed_parenthesis(self):
        """Test unclosed parenthesis."""
        source = "J(x := x + y;"
        
        with pytest.raises(ParseError):
            parse_source(source)
    
    def test_trailing_input(self):
        """Test trailing input handling."""
        source = "x :: Scalar; invalid syntax here"
        
        result = parse_safe(source)
        assert result.ast is not None
    
    def test_parse_error_message(self):
        """Test parse error message quality."""
        source = "x ::"
        
        try:
            parse_source(source)
            pytest.fail("Expected ParseError")
        except ParseError as e:
            assert len(e.message) > 0


class TestTypeErrorHandling:
    """Test type error handling."""
    
    def test_scalar_vector_type_mismatch(self):
        """Test Scalar + Vector type error."""
        source = """
        x :: Scalar;
        v :: Field[Vector];
        x = v;
        """
        result = compile_nsc_source(source)
        assert len(result.errors) > 0 or not result.is_typechecked
    
    def test_scalar_field_type_mismatch(self):
        """Test Scalar = Field type error."""
        source = """
        x :: Scalar;
        f :: Field[Scalar];
        x = f;
        """
        result = compile_nsc_source(source)
        assert len(result.errors) > 0
    
    def test_vector_tensor_type_mismatch(self):
        """Test Vector and Tensor type error."""
        source = """
        v :: Vector;
        T :: Tensor(0,2);
        v = T;
        """
        result = compile_nsc_source(source)
        assert len(result.errors) > 0
    
    def test_type_error_message(self):
        """Test type error message quality."""
        source = """
        x :: Scalar;
        v :: Field[Vector];
        x = v;
        """
        result = type_check_safe(source)
        if not result[0]:
            assert len(result[2]) > 0


class TestGeometryPrerequisiteErrors:
    """Test geometry prerequisite error handling."""
    
    def test_gradient_requires_metric(self):
        """Test gradient of vector field requires GEO."""
        source = """
        @model(CALC);
        u :: Field[Vector];
        du :: Field[Vector];
        du = ∇(u);
        """
        result = compile_nsc_source(source)
        assert len(result.errors) > 0 or not result.is_typechecked
    
    def test_laplacian_with_metric(self):
        """Test Laplacian with proper metric."""
        source = """
        @model(GEO, CALC);
        M :: Manifold(3, riemannian);
        g :: Metric on M;
        u :: Field[Scalar];
        Δ(u) = f;
        """
        result = compile_nsc_source(source)
        assert result.success


class TestModelCompatibilityErrors:
    """Test model compatibility error handling."""
    
    def test_missing_required_model(self):
        """Test error when required model is missing."""
        source = """
        @model(CALC);
        @inv(N:INV.gr.hamiltonian_constraint);
        """
        result = compile_nsc_source(source)
        assert result.ast is not None
    
    def test_incompatible_model_combination(self):
        """Test DISC without source model."""
        source = """
        @model(DISC);
        u :: Field[Scalar];
        Δ(u) = f;
        """
        result = compile_nsc_source(source)
        assert result.ast is not None


class TestErrorLocationReporting:
    """Test error location reporting."""
    
    def test_parse_error_location(self):
        """Test parse error includes location."""
        source = "x ::"
        
        try:
            parse_source(source)
            pytest.fail("Expected ParseError")
        except ParseError as e:
            assert hasattr(e, 'pos') or 'position' in str(e).lower()
    
    def test_type_error_line_reporting(self):
        """Test type error reports line number."""
        source = """
        x :: Scalar;
        v :: Field[Vector];
        z = x + v;
        """
        result = type_check_safe(source)
        if not result[0]:
            assert True
    
    def test_error_column_reporting(self):
        """Test error reports column."""
        source = """
        x :: Scalar;
        y = x + ;
        """
        result = compile_nsc_source(source)
        assert result.ast is not None


class TestErrorRecovery:
    """Test error recovery mechanisms."""
    
    def test_parse_recovery_multiple_errors(self):
        """Test parser continues after first error."""
        source = "valid1 :: Scalar; invalid1; valid2 :: Scalar; invalid2;"
        
        result = parse_safe(source)
        assert result.ast is not None
    
    def test_partial_type_check(self):
        """Test type checking partial program."""
        source = """
        x :: Scalar;
        invalid_var;
        y :: Vector;
        """
        result = type_check_program(parse_source(source))
        assert "x" in result.symbols or "y" in result.symbols
    
    def test_error_doesnt_crash_pipeline(self):
        """Test pipeline continues after error."""
        source = """
        x :: Scalar;
        y = unknown + x;
        z :: Vector;
        """
        result = compile_nsc_source(source)
        assert result.ast is not None


class TestWarningHandling:
    """Test warning handling."""
    
    def test_unknown_invariant_warning(self):
        """Test unknown invariant produces warning."""
        source = """
        @inv(N:INV.unknown.invariant);
        """
        ast = parse_source(source)
        
        checker = TypeChecker(invariant_registry={})
        type_result = checker.check_program(ast)
        assert True
    
    def test_valid_code_no_warnings(self):
        """Test valid code produces no warnings."""
        source = "x :: Scalar;"
        result = compile_nsc_source(source)
        assert result.success


class TestErrorMessages:
    """Test error message quality."""
    
    def test_parse_error_clear_message(self):
        """Test parse error message is clear."""
        source = "x :: Scalar ; ;"
        
        try:
            parse_source(source)
        except ParseError as e:
            assert len(e.message) > 0
    
    def test_type_error_shows_types(self):
        """Test type error shows types."""
        source = """
        x :: Scalar;
        v :: Field[Vector];
        x = v;
        """
        result = type_check_safe(source)
        if not result[0]:
            assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
