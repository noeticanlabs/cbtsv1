"""
NSC-M3L Type Checker + DISC Integration Tests
"""

import pytest

from tests.nsc_test_utils import (
    parse_source, type_check_program, compile_nsc_source,
    create_test_grid, create_test_quadrature, lower_to_disc
)
from src.nsc.type_checker import TypeChecker
from src.nsc.types import Scalar, Vector, Tensor, Field as FieldType
from src.nsc.ast import Model, Directive, DirectiveType
from src.nsc.disc_types import (
    Grid, FEMSpace, Stencil, StencilType, FEMElementType, StabilityInfo
)
from src.nsc.disc_lower import DiscreteLowerer, LoweringContext
from src.nsc.quadrature import gauss_legendre_1, gauss_legendre_2, gauss_legendre_3


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
    
    def test_tensor_field_lowering(self):
        """Test lowering tensor field type."""
        source = "T :: Field[Tensor(0,2)];"
        result = compile_nsc_source(source)
        assert result.success


class TestDISCTypeConsistency:
    """Test DISC types are consistent."""
    
    def test_discrete_field_type(self):
        """Test discrete field type."""
        source = "u :: Field[Scalar];"
        disc_output = lower_to_disc(source, shape=(10, 10))
        assert disc_output is not None
    
    def test_vector_field_discrete_type(self):
        """Test vector field discrete type."""
        source = "u :: Field[Vector];"
        disc_output = lower_to_disc(source, shape=(10, 10))
        assert disc_output is not None


class TestGridIntegration:
    """Test grid creation."""
    
    def test_1d_grid_lowering(self):
        """Test 1D grid."""
        source = "u :: Field[Scalar];"
        disc_output = lower_to_disc(source, shape=(100,))
        assert disc_output is not None
    
    def test_2d_grid_lowering(self):
        """Test 2D grid."""
        source = "u :: Field[Scalar];"
        disc_output = lower_to_disc(source, shape=(50, 50))
        assert disc_output is not None
        assert "grid_shape" in disc_output
    
    def test_3d_grid_lowering(self):
        """Test 3D grid."""
        source = "u :: Field[Scalar];"
        disc_output = lower_to_disc(source, shape=(30, 30, 30))
        assert disc_output is not None


class TestQuadratureIntegration:
    """Test quadrature integration."""
    
    def test_gauss_legendre_1_integration(self):
        """Test 1-point Gauss-Legendre."""
        source = "u :: Field[Scalar];"
        grid = create_test_grid(2, (50, 50))
        quad = gauss_legendre_1()
        context = LoweringContext(grid=grid, quadrature=quad)
        lowerer = DiscreteLowerer(context)
        disc_output = lowerer.lower_to_disc(parse_source(source))
        assert disc_output is not None
    
    def test_gauss_legendre_2_integration(self):
        """Test 2-point Gauss-Legendre."""
        source = "u :: Field[Scalar];"
        grid = create_test_grid(2, (50, 50))
        quad = gauss_legendre_2()
        context = LoweringContext(grid=grid, quadrature=quad)
        lowerer = DiscreteLowerer(context)
        disc_output = lowerer.lower_to_disc(parse_source(source))
        assert disc_output is not None
    
    def test_gauss_legendre_3_integration(self):
        """Test 3-point Gauss-Legendre."""
        source = "u :: Field[Scalar];"
        grid = create_test_grid(2, (50, 50))
        quad = gauss_legendre_3()
        context = LoweringContext(grid=grid, quadrature=quad)
        lowerer = DiscreteLowerer(context)
        disc_output = lowerer.lower_to_disc(parse_source(source))
        assert disc_output is not None


class TestStencilGeneration:
    """Test stencil generation."""
    
    def test_gradient_stencil_1d(self):
        """Test 1D gradient stencil."""
        grid = create_test_grid(1, (100,))
        quad = create_test_quadrature(2)
        context = LoweringContext(grid=grid, quadrature=quad)
        lowerer = DiscreteLowerer(context)
        stencils = lowerer.lower_gradient(None, grid)
        assert len(stencils) == 1
    
    def test_gradient_stencil_2d(self):
        """Test 2D gradient stencil."""
        grid = create_test_grid(2, (50, 50))
        quad = create_test_quadrature(2)
        context = LoweringContext(grid=grid, quadrature=quad)
        lowerer = DiscreteLowerer(context)
        stencils = lowerer.lower_gradient(None, grid)
        assert len(stencils) == 2
    
    def test_gradient_stencil_3d(self):
        """Test 3D gradient stencil."""
        grid = create_test_grid(3, (30, 30, 30))
        quad = create_test_quadrature(2)
        context = LoweringContext(grid=grid, quadrature=quad)
        lowerer = DiscreteLowerer(context)
        stencils = lowerer.lower_gradient(None, grid)
        assert len(stencils) == 3
    
    def test_divergence_stencil(self):
        """Test divergence stencil."""
        grid = create_test_grid(2, (50, 50))
        quad = create_test_quadrature(2)
        context = LoweringContext(grid=grid, quadrature=quad)
        lowerer = DiscreteLowerer(context)
        stencils = lowerer.lower_divergence(None, grid)
        assert len(stencils) == 2
    
    def test_laplacian_stencil(self):
        """Test Laplacian stencil."""
        grid = create_test_grid(2, (50, 50))
        quad = create_test_quadrature(2)
        context = LoweringContext(grid=grid, quadrature=quad)
        lowerer = DiscreteLowerer(context)
        stencil = lowerer.lower_laplacian(None, grid)
        assert stencil is not None
    
    def test_curl_stencil_2d(self):
        """Test 2D curl stencil."""
        grid = create_test_grid(2, (50, 50))
        quad = create_test_quadrature(2)
        context = LoweringContext(grid=grid, quadrature=quad)
        lowerer = DiscreteLowerer(context)
        stencils = lowerer.lower_curl(None, grid)
        assert len(stencils) == 1
    
    def test_curl_stencil_3d(self):
        """Test 3D curl stencil."""
        grid = create_test_grid(3, (30, 30, 30))
        quad = create_test_quadrature(2)
        context = LoweringContext(grid=grid, quadrature=quad)
        lowerer = DiscreteLowerer(context)
        stencils = lowerer.lower_curl(None, grid)
        assert len(stencils) == 3


class TestTimeDerivativeLowering:
    """Test time derivative lowering."""
    
    def test_forward_euler_lowering(self):
        """Test forward Euler."""
        grid = create_test_grid(1, (100,))
        quad = create_test_quadrature(1)
        context = LoweringContext(grid=grid, quadrature=quad)
        lowerer = DiscreteLowerer(context)
        stencil = lowerer.lower_time_derivative(None, scheme="forward_euler")
        assert stencil.accuracy == 1
    
    def test_backward_euler_lowering(self):
        """Test backward Euler."""
        grid = create_test_grid(1, (100,))
        quad = create_test_quadrature(1)
        context = LoweringContext(grid=grid, quadrature=quad)
        lowerer = DiscreteLowerer(context)
        stencil = lowerer.lower_time_derivative(None, scheme="backward_euler")
        assert stencil.accuracy == 1
    
    def test_crank_nicolson_lowering(self):
        """Test Crank-Nicolson."""
        grid = create_test_grid(1, (100,))
        quad = create_test_quadrature(1)
        context = LoweringContext(grid=grid, quadrature=quad)
        lowerer = DiscreteLowerer(context)
        stencil = lowerer.lower_time_derivative(None, scheme="crank_nicolson")
        assert stencil.accuracy == 2


class TestFEMIntegration:
    """Test FEM integration."""
    
    def test_lagrange_space(self):
        """Test Lagrange FEM space."""
        grid = create_test_grid(2, (50, 50))
        quad = create_test_quadrature(3)
        context = LoweringContext(grid=grid, quadrature=quad)
        lowerer = DiscreteLowerer(context)
        space = lowerer.create_fem_space("lagrange", degree=2, dim=2)
        assert space.element_type == FEMElementType.LAGRANGE
    
    def test_dg_space(self):
        """Test DG space."""
        grid = create_test_grid(2, (50, 50))
        quad = create_test_quadrature(3)
        context = LoweringContext(grid=grid, quadrature=quad)
        lowerer = DiscreteLowerer(context)
        space = lowerer.create_fem_space("dg", degree=1, dim=2)
        assert space.element_type == FEMElementType.DISCONTINUOUS_GALERKIN
    
    def test_mass_matrix_computation(self):
        """Test mass matrix."""
        grid = create_test_grid(1, (100,))
        quad = create_test_quadrature(2)
        context = LoweringContext(grid=grid, quadrature=quad)
        lowerer = DiscreteLowerer(context)
        space = FEMSpace(element_type=FEMElementType.LAGRANGE, degree=1, dim=100)
        M = lowerer.compute_mass_matrix(space)
        assert M.shape == (100, 100)
    
    def test_stiffness_matrix_computation(self):
        """Test stiffness matrix."""
        grid = create_test_grid(1, (100,))
        quad = create_test_quadrature(2)
        context = LoweringContext(grid=grid, quadrature=quad)
        lowerer = DiscreteLowerer(context)
        space = FEMSpace(element_type=FEMElementType.LAGRANGE, degree=1, dim=100)
        K = lowerer.compute_stiffness_matrix(space)
        assert K.shape == (100, 100)


class TestStabilityIntegration:
    """Test stability analysis."""
    
    def test_stability_info_generation(self):
        """Test stability info generation."""
        source = "@model(CALC, DISC);\nu :: Field[Scalar];"
        result = compile_nsc_source(source, target_models={Model.DISC})
        assert result.success
        assert result.stability_info is not None
    
    def test_stable_operator(self):
        """Test stability check."""
        grid = create_test_grid(2, (50, 50))
        quad = create_test_quadrature(2)
        context = LoweringContext(grid=grid, quadrature=quad)
        lowerer = DiscreteLowerer(context)
        stencil = lowerer.lower_laplacian(None, grid)
        stability = lowerer.analyze_operator_stability(stencil, grid)
        assert isinstance(stability, StabilityInfo)


class TestDISCOutputStructure:
    """Test DISC output structure."""
    
    def test_operator_dict(self):
        """Test operator dict."""
        source = "@model(CALC, DISC);\nu :: Field[Scalar];\nΔ(u) = f;"
        result = compile_nsc_source(source, target_models={Model.DISC})
        assert result.success
        assert result.disc_output is not None
        assert "operators" in result.disc_output
    
    def test_grid_info(self):
        """Test grid info."""
        source = "@model(DISC);\nu :: Field[Scalar];"
        result = compile_nsc_source(source, target_models={Model.DISC})
        assert result.success
        assert "grid_shape" in result.disc_output
    
    def test_quadrature_info(self):
        """Test quadrature info."""
        source = "@model(DISC);\nu :: Field[Scalar];"
        result = compile_nsc_source(source, target_models={Model.DISC})
        assert result.success
        assert "quadrature_degree" in result.disc_output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
