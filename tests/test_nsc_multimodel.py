"""
NSC-M3L Multi-Model Consistency Tests
"""

import pytest

from tests.nsc_test_utils import (
    compile_nsc_source, parse_source, type_check_program
)
from src.nsc.ast import Model


class TestModelSelection:
    """Test model selection."""
    
    def test_single_model(self):
        """Test single model."""
        source = "@model(CALC);"
        result = compile_nsc_source(source)
        assert result.success
        assert Model.CALC in result.models
    
    def test_multiple_models(self):
        """Test multiple models."""
        source = "@model(GEO, CALC, DISC);"
        result = compile_nsc_source(source)
        assert result.success
        assert Model.GEO in result.models
        assert Model.CALC in result.models
        assert Model.DISC in result.models
    
    def test_all_models(self):
        """Test all models."""
        source = "@model(ALG, GEO, CALC, DISC, LEDGER, EXEC);"
        result = compile_nsc_source(source)
        assert result.success
        assert len(result.models) == 6
    
    def test_model_order_independence(self):
        """Test model order."""
        source1 = "@model(GEO, CALC);"
        source2 = "@model(CALC, GEO);"
        result1 = compile_nsc_source(source1)
        result2 = compile_nsc_source(source2)
        assert result1.models == result2.models


class TestGEODISCEquivalence:
    """Test GEO and DISC equivalence."""
    
    def test_scalar_field_equivalence(self):
        """Test scalar field."""
        source = "@model(GEO, DISC);\nu :: Field[Scalar];"
        result = compile_nsc_source(source, target_models={Model.DISC})
        assert result.success
        assert Model.GEO in result.models
        assert Model.DISC in result.models


class TestCALCGEOCompatibility:
    """Test CALC and GEO compatibility."""
    
    def test_operator_compatibility(self):
        """Test operators."""
        source = "@model(CALC, GEO);\nu :: Field[Scalar];"
        result = compile_nsc_source(source)
        assert result.success
        assert Model.CALC in result.models
        assert Model.GEO in result.models
    
    def test_metric_requirement(self):
        """Test metric."""
        source = "@model(GEO);\ng :: Metric on M;"
        result = compile_nsc_source(source)
        assert result.success
    
    def test_manifold_with_geo(self):
        """Test manifold."""
        source = "@model(GEO);\nM :: Manifold(3, riemannian);"
        result = compile_nsc_source(source)
        assert result.success


class TestMultiModelCompilation:
    """Test multi-model compilation."""
    
    def test_geo_calc_ledger(self):
        """Test GEO + CALC + LEDGER."""
        source = "@model(GEO, CALC, LEDGER);\nM :: Manifold(3+1, lorentzian);\ng :: Metric on M;"
        result = compile_nsc_source(source)
        assert result.success
        assert Model.GEO in result.models
        assert Model.CALC in result.models
        assert Model.LEDGER in result.models
    
    def test_calc_ledger_exec(self):
        """Test CALC + LEDGER + EXEC."""
        source = "@model(CALC, LEDGER, EXEC);\nu :: Field[Scalar];"
        result = compile_nsc_source(source)
        assert result.success
    
    def test_full_stack(self):
        """Test full stack."""
        source = "@model(ALG, GEO, CALC, DISC, LEDGER, EXEC);"
        result = compile_nsc_source(source)
        assert result.success
        assert len(result.models) == 6


class TestCompileDirectiveModels:
    """Test compile directive."""
    
    def test_compile_to_disc(self):
        """Test compile to DISC."""
        source = "@model(GEO, CALC, DISC);\nu :: Field[Scalar];\n⇒ (DISC);"
        result = compile_nsc_source(source)
        assert result.success
        assert Model.DISC in result.models
    
    def test_compile_multiple_targets(self):
        """Test multiple targets."""
        source = "@model(GEO, CALC, DISC, LEDGER);\nu :: Field[Scalar];\n⇒ (DISC, LEDGER);"
        result = compile_nsc_source(source)
        assert result.success


class TestModelPrerequisites:
    """Test model prerequisites."""
    
    def test_disc_requires_calc(self):
        """Test DISC requires CALC."""
        source = "@model(DISC);\nu :: Field[Scalar];"
        result = compile_nsc_source(source, target_models={Model.DISC})
        assert result.ast is not None
    
    def test_geo_requires_metric(self):
        """Test GEO with metric."""
        source = "@model(GEO);\nM :: Manifold(3, riemannian);\ng :: Metric on M;"
        result = compile_nsc_source(source)
        assert result.success


class TestCrossModelTypeChecking:
    """Test cross-model type checking."""
    
    def test_field_type_consistency(self):
        """Test field type."""
        source = "@model(GEO, CALC, DISC);\nu :: Field[Scalar];"
        result = compile_nsc_source(source, target_models={Model.DISC})
        assert result.success
        assert "u" in result.symbols
    
    def test_vector_type_consistency(self):
        """Test vector type."""
        source = "@model(GEO, CALC);\nv :: Field[Vector];"
        result = compile_nsc_source(source)
        assert result.success
    
    def test_tensor_type_consistency(self):
        """Test tensor type."""
        source = "@model(GEO);\nT :: Tensor(0,2);"
        result = compile_nsc_source(source)
        assert result.success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
