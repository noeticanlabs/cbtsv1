"""
NSC-M3L Parser + Type Checker Integration Tests
"""

import pytest

from tests.nsc_test_utils import (
    parse_source, type_check_program, parse_safe, type_check_safe
)
from src.nsc.parse import ParseError


class TestParserOutputToTypeChecker:
    """Test parser output feeds type checker."""
    
    def test_scalar_decl_feeds_typechecker(self):
        """Verify scalar declaration is valid for type checker."""
        source = "x :: Scalar;"
        ast = parse_source(source)
        assert len(ast.statements) == 1
        
        result = type_check_program(ast)
        assert result.success
        assert "x" in result.symbols
    
    def test_vector_decl_feeds_typechecker(self):
        """Verify vector declaration is valid for type checker."""
        source = "v :: Vector;"
        ast = parse_source(source)
        result = type_check_program(ast)
        assert result.success
        assert "v" in result.symbols
    
    def test_tensor_decl_feeds_typechecker(self):
        """Verify tensor declaration is valid for type checker."""
        source = "T :: Tensor(0,2);"
        ast = parse_source(source)
        result = type_check_program(ast)
        assert result.success
        assert "T" in result.symbols
    
    def test_field_decl_feeds_typechecker(self):
        """Verify field declaration is valid for type checker."""
        source = "u :: Field[Scalar];"
        ast = parse_source(source)
        result = type_check_program(ast)
        assert result.success
        assert "u" in result.symbols
    
    def test_equation_feeds_typechecker(self):
        """Verify equation is valid for type checker."""
        source = "x :: Scalar;\ny :: Scalar;\nz = x + y;"
        ast = parse_source(source)
        result = type_check_program(ast)
        assert result.success
    
    def test_functional_feeds_typechecker(self):
        """Verify functional is valid for type checker."""
        source = "J(u :: Scalar) := u * u;"
        ast = parse_source(source)
        result = type_check_program(ast)
        assert result.success


class TestTypeCheckerErrorsFromParser:
    """Test parser errors don't crash type checker."""
    
    def test_parse_error_raises(self):
        """Ensure parser errors are properly raised."""
        source = "x ::"  # Missing type
        with pytest.raises(ParseError):
            parse_source(source)
    
    def test_parse_error_with_directive(self):
        """Test directive parsing errors."""
        source = "@model(GEO,);"  # Trailing comma
        ast = parse_source(source)
        assert ast is not None
    
    def test_type_error_after_valid_parse(self):
        """Test type errors occur after valid parse."""
        source = "x :: Scalar;\nv :: Vector;\nx = v;"
        ast = parse_source(source)
        assert ast is not None
        
        result = type_check_program(ast)
        assert not result.success


class TestASTTypeConsistency:
    """Test AST types are correctly interpreted."""
    
    def test_scalar_type_annotation(self):
        """Test Scalar type annotation."""
        source = "x :: Scalar;"
        result = type_check_safe(source)
        assert result[0]
    
    def test_vector_type_annotation(self):
        """Test Vector type annotation."""
        source = "v :: Vector;"
        result = type_check_safe(source)
        assert result[0]
    
    def test_tensor_type_annotation(self):
        """Test Tensor type annotation."""
        source = "T :: Tensor(1,1);"
        result = type_check_safe(source)
        assert result[0]
    
    def test_field_type_annotation(self):
        """Test Field type annotation."""
        source = "u :: Field[Vector];"
        result = type_check_safe(source)
        assert result[0]
    
    def test_nested_field_type(self):
        """Test nested Field type annotation."""
        source = "T :: Field[Tensor(0,2)];"
        result = type_check_safe(source)
        assert result[0]


class TestSymbolTableIntegration:
    """Test symbol table operations."""
    
    def test_symbol_registration(self):
        """Test symbols are registered."""
        source = "x :: Scalar;\ny :: Vector;\nz :: Tensor(0,2);"
        ast = parse_source(source)
        result = type_check_program(ast)
        
        assert "x" in result.symbols
        assert "y" in result.symbols
        assert "z" in result.symbols
    
    def test_symbol_lookup(self):
        """Test symbol lookup."""
        source = "my_field :: Field[Scalar];"
        ast = parse_source(source)
        result = type_check_program(ast)
        
        symbol = result.symbols.get("my_field")
        assert symbol is not None


class TestDirectiveIntegration:
    """Test directive parsing."""
    
    def test_model_directive_symbols(self):
        """Test @model directive."""
        source = "@model(GEO, CALC);"
        ast = parse_source(source)
        result = type_check_program(ast)
        assert result.success
    
    def test_inv_directive_parsing(self):
        """Test @inv directive parsing."""
        source = "@inv(N:INV.gr.hamiltonian_constraint);"
        ast = parse_source(source)
        assert ast is not None
    
    def test_compile_directive_parsing(self):
        """Test ⇒ directive parsing."""
        source = "⇒ (DISC);"
        ast = parse_source(source)
        assert ast is not None


class TestMultiStatementIntegration:
    """Test multiple statements."""
    
    def test_multiple_declarations(self):
        """Test multiple declarations."""
        source = "a :: Scalar;\nb :: Scalar;\nc :: Scalar;"
        ast = parse_source(source)
        result = type_check_program(ast)
        assert result.success
        assert len(result.symbols) >= 3
    
    def test_mixed_statements(self):
        """Test mixed declarations and equations."""
        source = "x :: Scalar;\ny :: Scalar;\nz :: Scalar;\nz = x + y;"
        ast = parse_source(source)
        result = type_check_program(ast)
        assert result.success
    
    def test_program_with_directives(self):
        """Test program with directives."""
        source = "@model(GEO, CALC);\n@inv(N:INV.gr.hamiltonian_constraint);"
        ast = parse_source(source)
        result = type_check_program(ast)
        assert result.success


class TestExpressionIntegration:
    """Test expression parsing and type checking."""
    
    def test_binary_expression(self):
        """Test binary expression."""
        source = "a :: Scalar;\nb :: Scalar;\nc = a + b;"
        result = type_check_program(parse_source(source))
        assert result.success
    
    def test_nested_expression(self):
        """Test nested expression."""
        source = "a :: Field[Scalar];\nb :: Field[Scalar];\nc :: Field[Scalar];\n(a + b) * c = d;"
        result = type_check_program(parse_source(source))
        assert result.success


class TestBindingIntegration:
    """Test binding integration."""
    
    def test_single_binding(self):
        """Test functional with single binding."""
        source = "J(x :: Scalar) := x * x;"
        result = type_check_program(parse_source(source))
        assert result.success
    
    def test_multiple_bindings(self):
        """Test functional with multiple bindings."""
        source = "J(x :: Scalar, y :: Scalar) := x + y;"
        result = type_check_program(parse_source(source))
        assert result.success


class TestPipelineRecovery:
    """Test pipeline recovery."""
    
    def test_parse_recovery_continues(self):
        """Test parser continues after error."""
        source = "valid1 :: Scalar; invalid; valid2 :: Scalar;"
        ast = parse_source(source)
        assert ast is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
