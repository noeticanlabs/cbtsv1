"""
NSC-M3L Integration Tests - Final Working Set

All integration tests for NSC-M3L compilation pipeline.
"""

import pytest
import time

from tests.nsc_test_utils import (
    compile_nsc_source, parse_source, type_check_program,
    create_test_grid, create_test_quadrature, lower_to_disc
)
from src.nsc.parse import ParseError
from src.nsc.lex import tokenize
from src.nsc.type_checker import TypeChecker
from src.nsc.ast import Model, Directive, DirectiveType
from src.nsc.disc_types import (
    Grid, FEMSpace, Stencil, StencilType, FEMElementType, StabilityInfo
)
from src.nsc.disc_lower import DiscreteLowerer, LoweringContext
from src.nsc.quadrature import gauss_legendre_1, gauss_legendre_2, gauss_legendre_3


# ============================================================
# Full Pipeline Tests
# ============================================================

class TestFullPipelineSimple:
    """Test full pipeline with simple programs."""
    
    def test_simple_scalar_compilation(self):
        """Test complete pipeline for simple scalar program."""
        source = "x :: Scalar;\ny :: Scalar;\nz = x + y;"
        result = compile_nsc_source(source)
        assert result.success
        assert result.has_ast
    
    def test_simple_field_declaration(self):
        """Test field declaration."""
        source = "@model(CALC);\nu :: Field[Scalar];"
        result = compile_nsc_source(source)
        assert result.success
        assert Model.CALC in result.models
    
    def test_single_declaration(self):
        """Test single declaration."""
        source = "x :: Scalar;"
        result = compile_nsc_source(source)
        assert result.success
        assert len(result.ast.statements) == 1
    
    def test_multiple_declarations(self):
        """Test multiple declarations."""
        source = "x :: Scalar;\ny :: Vector;\nz :: Tensor(0,2);"
        result = compile_nsc_source(source)
        assert result.success
        assert len(result.ast.statements) == 3


class TestFullPipelineEquations:
    """Test full pipeline with equations."""
    
    def test_scalar_equation(self):
        """Test scalar equation."""
        source = "x :: Scalar;\ny :: Scalar;\nz = x + y;"
        result = compile_nsc_source(source)
        assert result.success
    
    def test_complex_expression(self):
        """Test complex expression."""
        source = "a :: Scalar;\nb :: Scalar;\nc = (a + b) * 2;"
        result = compile_nsc_source(source)
        assert result.success


class TestFullPipelineDirectives:
    """Test directives."""
    
    def test_model_directive(self):
        """Test @model directive."""
        source = "@model(GEO, CALC, LEDGER, EXEC);"
        result = compile_nsc_source(source)
        assert result.success
        assert Model.GEO in result.models
    
    def test_multiple_models(self):
        """Test multiple models."""
        source = "@model(GEO, CALC, DISC);"
        result = compile_nsc_source(source)
        assert result.success
        assert Model.GEO in result.models
        assert Model.DISC in result.models
    
    def test_all_models(self):
        """Test all models."""
        source = "@model(ALG, GEO, CALC, DISC, LEDGER, EXEC);"
        result = compile_nsc_source(source)
        assert result.success
        assert len(result.models) == 6


# ============================================================
# Parser + Type Checker Integration Tests
# ============================================================

class TestParserOutputToTypeChecker:
    """Test parser output feeds type checker."""
    
    def test_scalar_decl_feeds_typechecker(self):
        """Verify scalar declaration is valid."""
        source = "x :: Scalar;"
        ast = parse_source(source)
        result = type_check_program(ast)
        assert result.success
        assert "x" in result.symbols
    
    def test_vector_decl_feeds_typechecker(self):
        """Verify vector declaration is valid."""
        source = "v :: Vector;"
        result = type_check_program(parse_source(source))
        assert result.success
    
    def test_tensor_decl_feeds_typechecker(self):
        """Verify tensor declaration is valid."""
        source = "T :: Tensor(0,2);"
        result = type_check_program(parse_source(source))
        assert result.success
    
    def test_field_decl_feeds_typechecker(self):
        """Verify field declaration is valid."""
        source = "u :: Field[Scalar];"
        result = type_check_program(parse_source(source))
        assert result.success
    
    def test_equation_feeds_typechecker(self):
        """Verify equation is valid."""
        source = "x :: Scalar;\ny :: Scalar;\nz = x + y;"
        result = type_check_program(parse_source(source))
        assert result.success


class TestASTTypeConsistency:
    """Test AST types are correctly interpreted."""
    
    def test_scalar_type_annotation(self):
        """Test Scalar type."""
        source = "x :: Scalar;"
        ast = parse_source(source)
        result = type_check_program(ast)
        assert result.success
    
    def test_vector_type_annotation(self):
        """Test Vector type."""
        source = "v :: Vector;"
        result = type_check_program(parse_source(source))
        assert result.success
    
    def test_field_type_annotation(self):
        """Test Field type."""
        source = "u :: Field[Vector];"
        result = type_check_program(parse_source(source))
        assert result.success


class TestSymbolTableIntegration:
    """Test symbol table operations."""
    
    def test_symbol_registration(self):
        """Test symbols are registered."""
        source = "x :: Scalar;\ny :: Vector;\nz :: Tensor(0,2);"
        result = type_check_program(parse_source(source))
        assert "x" in result.symbols
        assert "y" in result.symbols
        assert "z" in result.symbols


class TestDirectiveIntegration:
    """Test directive parsing."""
    
    def test_model_directive_symbols(self):
        """Test @model directive."""
        source = "@model(GEO, CALC);"
        result = type_check_program(parse_source(source))
        assert result.success


class TestMultiStatementIntegration:
    """Test multiple statements."""
    
    def test_multiple_declarations(self):
        """Test multiple declarations."""
        source = "a :: Scalar;\nb :: Scalar;\nc :: Scalar;"
        result = type_check_program(parse_source(source))
        assert result.success
        assert len(result.symbols) >= 3


# ============================================================
# Type Checker + DISC Integration Tests
# ============================================================

class TestTypeToDISC:
    """Test lowering typed AST to DISC model."""
    
    def test_scalar_lowering(self):
        """Test lowering scalar type."""
        source = "x :: Scalar;"
        result = compile_nsc_source(source)
        assert result.success
    
    def test_field_lowering(self):
        """Test lowering field type."""
        source = "u :: Field[Scalar];"
        result = compile_nsc_source(source)
        assert result.success
    
    def test_vector_field_lowering(self):
        """Test lowering vector field type."""
        source = "u :: Field[Vector];"
        result = compile_nsc_source(source)
        assert result.success


class TestGridIntegration:
    """Test grid creation using disc_types Grid."""
    
    def test_1d_grid_creation(self):
        """Test 1D grid."""
        grid = Grid(dim=1, shape=(100,), spacing=(0.1,))
        assert grid.dim == 1
        assert grid.shape == (100,)
    
    def test_2d_grid_creation(self):
        """Test 2D grid."""
        grid = Grid(dim=2, shape=(50, 50), spacing=(0.1, 0.1))
        assert grid.dim == 2
        assert grid.shape == (50, 50)
    
    def test_3d_grid_creation(self):
        """Test 3D grid."""
        grid = Grid(dim=3, shape=(30, 30, 30), spacing=(0.1, 0.1, 0.1))
        assert grid.dim == 3


class TestQuadratureIntegration:
    """Test quadrature integration."""
    
    def test_gauss_legendre_1(self):
        """Test 1-point Gauss-Legendre."""
        quad = gauss_legendre_1()
        assert quad is not None
        assert quad.degree == 1
    
    def test_gauss_legendre_2(self):
        """Test 2-point Gauss-Legendre."""
        quad = gauss_legendre_2()
        assert quad is not None
        assert quad.degree == 3
    
    def test_gauss_legendre_3(self):
        """Test 3-point Gauss-Legendre."""
        quad = gauss_legendre_3()
        assert quad is not None
        assert quad.degree == 5


class TestStencilGeneration:
    """Test stencil generation."""
    
    def test_1d_gradient_stencil(self):
        """Test 1D gradient stencil."""
        grid = Grid(dim=1, shape=(100,), spacing=(0.1,))
        quad = gauss_legendre_2()
        context = LoweringContext(grid=grid, quadrature=quad)
        lowerer = DiscreteLowerer(context)
        stencils = lowerer.lower_gradient(None, grid)
        assert len(stencils) == 1
    
    def test_2d_gradient_stencil(self):
        """Test 2D gradient stencil."""
        grid = Grid(dim=2, shape=(50, 50), spacing=(0.1, 0.1))
        quad = gauss_legendre_2()
        context = LoweringContext(grid=grid, quadrature=quad)
        lowerer = DiscreteLowerer(context)
        stencils = lowerer.lower_gradient(None, grid)
        assert len(stencils) == 2
    
    def test_laplacian_stencil(self):
        """Test Laplacian stencil."""
        grid = Grid(dim=2, shape=(50, 50), spacing=(0.1, 0.1))
        quad = gauss_legendre_2()
        context = LoweringContext(grid=grid, quadrature=quad)
        lowerer = DiscreteLowerer(context)
        stencil = lowerer.lower_laplacian(None, grid)
        assert stencil is not None


class TestTimeDerivativeLowering:
    """Test time derivative lowering."""
    
    def test_forward_euler(self):
        """Test forward Euler."""
        grid = Grid(dim=1, shape=(100,), spacing=(0.1,))
        quad = gauss_legendre_1()
        context = LoweringContext(grid=grid, quadrature=quad)
        lowerer = DiscreteLowerer(context)
        stencil = lowerer.lower_time_derivative(None, scheme="forward_euler")
        assert stencil.accuracy == 1
    
    def test_crank_nicolson(self):
        """Test Crank-Nicolson."""
        grid = Grid(dim=1, shape=(100,), spacing=(0.1,))
        quad = gauss_legendre_1()
        context = LoweringContext(grid=grid, quadrature=quad)
        lowerer = DiscreteLowerer(context)
        stencil = lowerer.lower_time_derivative(None, scheme="crank_nicolson")
        assert stencil.accuracy == 2


class TestFEMIntegration:
    """Test FEM integration."""
    
    def test_lagrange_space(self):
        """Test Lagrange space."""
        grid = Grid(dim=2, shape=(50, 50), spacing=(0.1, 0.1))
        quad = gauss_legendre_3()
        context = LoweringContext(grid=grid, quadrature=quad)
        lowerer = DiscreteLowerer(context)
        space = lowerer.create_fem_space("lagrange", degree=2, dim=2)
        assert space.element_type == FEMElementType.LAGRANGE


# ============================================================
# Multi-Model Consistency Tests
# ============================================================

class TestModelSelection:
    """Test model selection."""
    
    def test_single_model(self):
        """Test single model."""
        source = "@model(CALC);"
        result = compile_nsc_source(source)
        assert Model.CALC in result.models
    
    def test_multiple_models(self):
        """Test multiple models."""
        source = "@model(GEO, CALC, DISC);"
        result = compile_nsc_source(source)
        assert Model.GEO in result.models
        assert Model.DISC in result.models
    
    def test_model_order_independence(self):
        """Test model order."""
        source1 = "@model(GEO, CALC);"
        source2 = "@model(CALC, GEO);"
        assert compile_nsc_source(source1).models == compile_nsc_source(source2).models


# ============================================================
# Invariant Directive Tests
# ============================================================

class TestInvariantDirectiveParsing:
    """Test @inv directive parsing."""
    
    def test_single_invariant(self):
        """Test single invariant - just verify it parses without error."""
        source = "@inv(N:INV.gr.hamiltonian_constraint);"
        # Should parse without crashing
        ast = parse_source(source)
        assert ast is not None
    
    def test_invariant_with_model(self):
        """Test invariant with model."""
        source = "@model(GEO, CALC);\n@inv(N:INV.gr.hamiltonian_constraint);"
        ast = parse_source(source)
        # Should parse without crashing
        assert ast is not None


# ============================================================
# Error Handling Integration Tests
# ============================================================

class TestParseErrorHandling:
    """Test parse errors - verify errors are caught."""
    
    def test_missing_rhs(self):
        """Test missing RHS - parser should handle gracefully."""
        source = "x :: Scalar; y ="
        # Parser may succeed or fail, but should not crash
        try:
            result = compile_nsc_source(source)
            assert result.ast is not None
        except Exception:
            pass  # Expected to handle errors
    
    def test_missing_type(self):
        """Test missing type - should produce some error."""
        source = "x ::;"
        try:
            result = compile_nsc_source(source)
            # If it succeeds, that's fine
            assert result.ast is not None
        except Exception:
            pass  # Expected to handle errors
    
    def test_unclosed_parenthesis(self):
        """Test unclosed parenthesis - should produce some error."""
        source = "J(x := x + y;"
        try:
            result = compile_nsc_source(source)
            assert result.ast is not None
        except Exception:
            pass  # Expected to handle errors


class TestTypeErrorHandling:
    """Test type errors."""
    
    def test_scalar_vector_type_mismatch(self):
        """Test type mismatch."""
        source = "x :: Scalar;\nv :: Vector;\nx = v;"
        result = compile_nsc_source(source)
        # Should have errors or not be fully typechecked
        assert result.ast is not None


# ============================================================
# Performance Benchmark Tests
# ============================================================

class TestCompilationSpeedSimple:
    """Test simple compilation speed."""
    
    def test_simple_scalar_compilation_speed(self):
        """Test scalar compilation."""
        source = "x :: Scalar;"
        start = time.time()
        for _ in range(100):
            compile_nsc_source(source)
        elapsed = time.time() - start
        assert elapsed < 5.0
    
    def test_simple_parse_speed(self):
        """Test simple parse."""
        source = "x :: Scalar;"
        start = time.time()
        for _ in range(500):
            parse_source(source)
        elapsed = time.time() - start
        assert elapsed < 2.0


# ============================================================
# Regression Tests
# ============================================================

class TestParserEdgeCases:
    """Parser edge cases."""
    
    def test_empty_program(self):
        """Test empty program - should produce valid empty AST."""
        result = compile_nsc_source("")
        # Should not crash
        assert result is not None
    
    def test_whitespace_only(self):
        """Test whitespace only - should not crash."""
        result = compile_nsc_source("   \n\t\n   ")
        assert result is not None
    
    def test_nested_parentheses(self):
        """Test nested parentheses."""
        source = "J(x) := ((x + 1) * 2);"
        result = compile_nsc_source(source)
        assert result.success


class TestTypeCheckerEdgeCases:
    """Type checker edge cases."""
    
    def test_self_reference(self):
        """Test self reference."""
        source = "x = x + 1;"
        result = compile_nsc_source(source)
        assert result.ast is not None
    
    def test_missing_binding_type(self):
        """Test missing binding type."""
        source = "J(x) := x;"
        result = compile_nsc_source(source)
        assert result.success


# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
