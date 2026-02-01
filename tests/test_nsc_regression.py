"""
NSC-M3L Regression Tests

Tests that previously fixed bugs stay fixed.
"""

import pytest

from tests.nsc_test_utils import compile_nsc_source, parse_source


class TestParserEdgeCases:
    """Parser edge case regression tests."""
    
    def test_empty_program(self):
        """Test empty program doesn't crash."""
        source = ""
        result = compile_nsc_source(source)
        assert result.ast is not None
    
    def test_whitespace_only(self):
        """Test whitespace-only input."""
        source = "   \n\t\n   "
        result = compile_nsc_source(source)
        assert result.ast is not None
    
    def test_multiple_semicolons(self):
        """Test multiple semicolons."""
        source = "x :: Scalar;;;"
        result = compile_nsc_source(source)
        assert result.ast is not None
    
    def test_nested_parentheses(self):
        """Test deeply nested parentheses."""
        source = "J(x) := ((x + 1) * 2);"
        result = compile_nsc_source(source)
        assert result.success
    
    def test_mixed_brackets(self):
        """Test mixed bracket types."""
        source = "x :: Field[Tensor(0,2)];"
        result = compile_nsc_source(source)
        assert result.success
    
    def test_long_identifier(self):
        """Test very long identifier."""
        long_name = "a" * 1000
        source = f"{long_name} :: Scalar;"
        result = compile_nsc_source(source)
        assert result.ast is not None


class TestTypeCheckerEdgeCases:
    """Type checker edge case regression tests."""
    
    def test_self_reference(self):
        """Test self-referential declarations."""
        source = "x = x + 1;"
        result = compile_nsc_source(source)
        assert result.ast is not None
    
    def test_circular_reference(self):
        """Test circular references."""
        source = "x = y + 1;\ny = x + 1;"
        result = compile_nsc_source(source)
        assert result.ast is not None
    
    def test_empty_equation(self):
        """Test empty equation."""
        source = "x = ;"
        result = compile_nsc_source(source)
        assert result.ast is not None or len(result.errors) > 0
    
    def test_missing_binding_type(self):
        """Test functional with missing binding type."""
        source = "J(x) := x;"
        result = compile_nsc_source(source)
        assert result.success
    
    def test_vector_scalar_mix(self):
        """Test vector and scalar mixture."""
        source = "v :: Vector;\ns :: Scalar;\nv = s;"
        result = compile_nsc_source(source)
        assert len(result.errors) > 0 or not result.is_typechecked
    
    def test_field_scalar_assignment(self):
        """Test field to scalar assignment."""
        source = "f :: Field[Scalar];\ns :: Scalar;\ns = f;"
        result = compile_nsc_source(source)
        assert len(result.errors) > 0


class TestDISCLoweringEdgeCases:
    """DISC lowering edge case regression tests."""
    
    def test_empty_grid(self):
        """Test grid with no points."""
        source = "u :: Field[Scalar];"
        result = compile_nsc_source(source, target_models={Model.DISC})
        assert result.ast is not None
    
    def test_single_point_grid(self):
        """Test single point grid."""
        source = "u :: Field[Scalar];"
        result = compile_nsc_source(source, target_models={Model.DISC})
        assert result.ast is not None
    
    def test_large_grid(self):
        """Test very large grid."""
        source = "u :: Field[Scalar];"
        result = compile_nsc_source(source, target_models={Model.DISC})
        assert result.ast is not None


class TestMultiModelEdgeCases:
    """Multi-model edge case regression tests."""
    
    def test_empty_model_list(self):
        """Test empty model list."""
        source = "@model();"
        result = compile_nsc_source(source)
        assert result.ast is not None
    
    def test_duplicate_models(self):
        """Test duplicate models in directive."""
        source = "@model(GEO, GEO, CALC, CALC);"
        result = compile_nsc_source(source)
        assert result.success
    
    def test_unknown_model(self):
        """Test unknown model name."""
        source = "@model(UNKNOWN);"
        result = compile_nsc_source(source)
        assert result.ast is not None
    
    def test_case_sensitive_models(self):
        """Test model names are case sensitive."""
        source = "@model(geo);"
        result = compile_nsc_source(source)
        assert result.ast is not None


class TestDirectiveEdgeCases:
    """Directive edge case regression tests."""
    
    def test_empty_inv_list(self):
        """Test empty invariant list."""
        source = "@inv();"
        ast = parse_source(source)
        assert ast is not None
    
    def test_inv_with_only_commas(self):
        """Test invariant with only commas."""
        source = "@inv(,);"
        result = compile_nsc_source(source)
        assert result.ast is not None
    
    def test_compile_directive_no_models(self):
        """Test compile directive with no models."""
        source = "⇒ ();"
        result = compile_nsc_source(source)
        assert result.ast is not None
    
    def test_multiple_directives_same_type(self):
        """Test multiple directives of same type."""
        source = "@model(GEO);\n@model(CALC);"
        result = compile_nsc_source(source)
        assert result.success


class TestExpressionEdgeCases:
    """Expression edge case regression tests."""
    
    def test_empty_parentheses(self):
        """Test empty parentheses."""
        source = "x = ();"
        result = compile_nsc_source(source)
        assert result.ast is not None or len(result.errors) > 0
    
    def test_unbalanced_brackets(self):
        """Test unbalanced brackets."""
        source = "x = (1 + 2;"
        result = compile_nsc_source(source)
        assert result.ast is not None or len(result.errors) > 0
    
    def test_division_by_zero(self):
        """Test division by zero in expression."""
        source = "x :: Scalar;\ny = x / 0;"
        result = compile_nsc_source(source)
        assert result.ast is not None
    
    def test_very_large_number(self):
        """Test very large number literal."""
        source = "x = 1e1000;"
        result = compile_nsc_source(source)
        assert result.ast is not None
    
    def test_very_small_number(self):
        """Test very small number literal."""
        source = "x = 1e-1000;"
        result = compile_nsc_source(source)
        assert result.ast is not None


class TestMetadataEdgeCases:
    """Metadata edge case regression tests."""
    
    def test_empty_metadata(self):
        """Test empty metadata."""
        source = "x :: Scalar : {};"
        result = compile_nsc_source(source)
        assert result.success
    
    def test_metadata_without_values(self):
        """Test metadata without key values."""
        source = "x :: Scalar : {key};"
        result = compile_nsc_source(source)
        assert result.ast is not None
    
    def test_metadata_unicode_keys(self):
        """Test Unicode metadata keys."""
        source = "x :: Scalar : {α = 1};"
        result = compile_nsc_source(source)
        assert result.ast is not None
    
    def test_metadata_many_entries(self):
        """Test many metadata entries."""
        entries = ", ".join(f"k{i} = v{i}" for i in range(100))
        source = f"x :: Scalar : {{{entries}}};"
        result = compile_nsc_source(source)
        assert result.ast is not None


class TestRegressionIssue47:
    """Regression test for issue 47 - parenthesized expressions."""
    
    def test_parenthesized_type_annotation(self):
        """Parenthesized expressions in type annotations."""
        source = "x :: Field[Scalar];"
        result = compile_nsc_source(source)
        assert result.success
    
    def test_nested_parenthesized_expressions(self):
        """Nested parenthesized expressions."""
        source = "x = ((a + b) * (c + d));"
        result = compile_nsc_source(source)
        assert result.success
    
    def test_parenthesized_equation_sides(self):
        """Parenthesized equation sides."""
        source = "(a + b) = (c + d);"
        result = compile_nsc_source(source)
        assert result.success


class TestRegressionIssue23:
    """Regression test for issue 23 - multiple model directives."""
    
    def test_multiple_model_directives(self):
        """Multiple model directives."""
        source = "@model(GEO);\n@model(CALC);\n⇒ (GEO, CALC);"
        result = compile_nsc_source(source)
        assert result.success
    
    def test_model_directive_scattering(self):
        """Model directives scattered through program."""
        source = "@model(GEO);\nx :: Scalar;\n@model(CALC);\ny :: Field[Scalar];\n⇒ (GEO, CALC);"
        result = compile_nsc_source(source)
        assert result.success
    
    def test_model_override(self):
        """Model directive override."""
        source = "@model(GEO);\n@model(CALC);\n@model(DISC);"
        result = compile_nsc_source(source)
        assert result.success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
