"""
NSC-M3L Full Pipeline Integration Tests

Tests the complete compilation pipeline from source code through all stages.
"""

import pytest

from tests.nsc_test_utils import (
    compile_nsc_source, parse_source, type_check_program,
    assert_no_errors
)
from src.nsc.ast import Model


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
        source = "a :: Field[Scalar];\nb :: Field[Scalar];\nc :: Field[Scalar];\n(a + b) * c = d;"
        result = compile_nsc_source(source)
        assert result.success


class TestFullPipelineGR:
    """Test full pipeline with GR programs."""
    
    def test_gr_model_directive(self):
        """Test GR model directive."""
        source = "@model(GEO, CALC, LEDGER);"
        result = compile_nsc_source(source)
        assert result.success
        assert Model.GEO in result.models
        assert Model.CALC in result.models
        assert Model.LEDGER in result.models
    
    def test_gr_with_manifolds(self):
        """Test GR with manifold declarations."""
        source = """
        @model(GEO, CALC, LEDGER);
        M :: Manifold(3+1, lorentzian);
        g :: Metric on M;
        """
        result = compile_nsc_source(source)
        assert result.success
    
    def test_gr_multimodel(self):
        """Test GR multi-model compilation."""
        source = "@model(GEO, CALC, DISC);\nu :: Field[Scalar];"
        result = compile_nsc_source(source, target_models={Model.DISC})
        assert result.success
        assert Model.GEO in result.models
        assert Model.DISC in result.models


class TestFullPipelineNS:
    """Test full pipeline with Navier-Stokes programs."""
    
    def test_ns_model(self):
        """Test NS model."""
        source = "@model(CALC, LEDGER, EXEC);"
        result = compile_nsc_source(source)
        assert result.success
    
    def test_ns_with_fields(self):
        """Test NS with field declarations."""
        source = """
        @model(CALC, LEDGER, EXEC);
        u :: Field[Vector];
        p :: Field[Scalar];
        """
        result = compile_nsc_source(source)
        assert result.success


class TestFullPipelineWaveEquation:
    """Test wave equation."""
    
    def test_wave_equation(self):
        """Test wave equation compilation."""
        source = "@model(GEO, CALC, DISC);\nu :: Field[Scalar];"
        result = compile_nsc_source(source, target_models={Model.DISC})
        assert result.success
        assert result.disc_output is not None


class TestFullPipelineDirectives:
    """Test full pipeline with directives."""
    
    def test_model_directive(self):
        """Test @model directive."""
        source = "@model(GEO, CALC, LEDGER, EXEC);"
        result = compile_nsc_source(source)
        assert result.success
        assert Model.GEO in result.models
    
    def test_inv_directive(self):
        """Test @inv directive."""
        source = """
        @model(GEO, CALC);
        @inv(N:INV.gr.hamiltonian_constraint);
        M :: Manifold(3+1, lorentzian);
        g :: Metric on M;
        """
        result = compile_nsc_source(source)
        assert result.ast is not None
    
    def test_compile_directive(self):
        """Test ⇒ directive."""
        source = "@model(GEO, DISC);\nu :: Field[Scalar];\n⇒ (DISC);"
        result = compile_nsc_source(source)
        assert result.success
        assert Model.DISC in result.models
    
    def test_multiple_directives(self):
        """Test multiple directives."""
        source = """
        @model(CALC, GEO, DISC);
        @inv(N:INV.gr.hamiltonian_constraint);
        u :: Field[Scalar];
        ⇒ (DISC);
        """
        result = compile_nsc_source(source)
        assert result.success


class TestFullPipelineMetadata:
    """Test metadata."""
    
    def test_declaration_with_metadata(self):
        """Test declaration with metadata."""
        source = "u :: Field[Scalar] : regularity = C2;"
        result = compile_nsc_source(source)
        assert result.success


class TestFullPipelineErrorHandling:
    """Test error handling."""
    
    def test_syntax_error_recovery(self):
        """Test parser handles trailing input."""
        source = "x :: Scalar; extra invalid token"
        result = compile_nsc_source(source)
        assert result.ast is not None


class TestFullPipelineComplex:
    """Test complex programs."""
    
    def test_multimodel_with_all_models(self):
        """Test all models."""
        source = "@model(ALG, GEO, CALC, DISC, LEDGER, EXEC);"
        result = compile_nsc_source(source)
        assert result.success
        assert len(result.models) >= 2
    
    def test_nested_operators(self):
        """Test nested operators."""
        source = "@model(CALC);\nu :: Field[Scalar];\ngrad(u) = g;"
        result = compile_nsc_source(source)
        assert result.success
    
    def test_chained_equations(self):
        """Test chained equations."""
        source = """
        @model(CALC);
        a :: Field[Scalar];
        b :: Field[Scalar];
        c :: Field[Scalar];
        a + b = c;
        """
        result = compile_nsc_source(source)
        assert result.success


class TestPipelineTiming:
    """Test timing."""
    
    def test_simple_compilation_time(self):
        """Test simple compilation is fast."""
        import time
        source = "x :: Scalar;"
        start = time.time()
        for _ in range(10):
            compile_nsc_source(source)
        elapsed = time.time() - start
        assert elapsed < 5.0
    
    def test_complex_compilation_time(self):
        """Test complex compilation."""
        import time
        source = "@model(GEO, CALC, LEDGER);\nM :: Manifold(3+1, lorentzian);\ng :: Metric on M;"
        start = time.time()
        compile_nsc_source(source)
        elapsed = time.time() - start
        assert elapsed < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
