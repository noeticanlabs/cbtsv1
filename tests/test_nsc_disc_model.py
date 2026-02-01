"""
NSC-M3L DISC Model Tests

Comprehensive tests for the DISC (Discrete Numerics) Model implementation.
Tests cover:
- Grid creation and indexing
- Stencil generation and application
- Quadrature rule integration
- Stability analysis
- Lowering from GEO/CALC to DISC
- CFL condition checking
- Discrete operator composition
- Multi-dimensional discretizations
"""

import pytest
import numpy as np
from typing import Tuple

from src.nsc.disc_types import (
    UnstructuredGrid,
    FEMSpace, FDSpace, LatticeSpace,
    Stencil, QuadratureRule, InterpolationMatrix,
    DiscreteField, DiscreteOperator, StabilityInfo,
    StencilType, BoundaryConditionType, FEMElementType, LatticeType
)
from src.nsc.stencils import (
    stencil_1d_central_2, stencil_1d_forward_1, stencil_1d_backward_1,
    stencil_1d_central_4, stencil_1d_second_central_2,
    stencil_5_point, stencil_9_point, stencil_7_point_3d,
    stencil_gradient_2d_central, stencil_divergence_2d_central, stencil_curl_2d,
    stencil_gradient_nd, stencil_divergence_nd,
    stencil_isotropic_laplacian,
    apply_stencil_field, compose_stencils, scale_stencil, add_stencils
)
from src.nsc.quadrature import (
    gauss_legendre_1, gauss_legendre_2, gauss_legendre_3,
    trapezoidal_rule, simpson_rule,
    quadrature_triangle_1, quadrature_triangle_3,
    quadrature_tetrahedron_1,
    gauss_legendre_tensor_2d, gauss_legendre_tensor_3d,
    integrate, integrate_function
)
from src.nsc.stability import (
    check_cfl_condition, compute_spectral_radius,
    check_lax_equivalence, estimate_condition_number,
    analyze_stability, check_time_step_stability
)
from src.nsc.grid import (
    Grid as GridType, UnstructuredGrid as UnstructuredGridType,
    create_uniform_grid, create_staggered_grid
)
from src.nsc.disc_lower import (
    DiscreteLowerer, LoweringContext
)


class TestGridCreation:
    """Test grid creation and basic properties."""
    
    def test_1d_grid_creation(self):
        """Test 1D grid creation."""
        grid = GridType(shape=(10,), spacing=(0.1,), origin=(0.0,))
        
        assert grid.dim == 1
        assert grid.shape == (10,)
        assert grid.spacing == (0.1,)
        assert grid.num_points == 10
        assert grid.cell_volume == 0.1
    
    def test_2d_grid_creation(self):
        """Test 2D grid creation."""
        grid = GridType(shape=(10, 20), spacing=(0.1, 0.05), origin=(0.0, 0.0))
        
        assert grid.dim == 2
        assert grid.shape == (10, 20)
        assert grid.spacing == (0.1, 0.05)
        assert grid.num_points == 200
        assert abs(grid.cell_volume - 0.005) < 1e-10
    
    def test_3d_grid_creation(self):
        """Test 3D grid creation."""
        grid = GridType(shape=(10, 20, 30), spacing=(0.1, 0.1, 0.1))
        
        assert grid.dim == 3
        assert grid.shape == (10, 20, 30)
        assert grid.num_points == 6000
        assert abs(grid.cell_volume - 0.001) < 1e-10
    
    def test_grid_index_to_coord(self):
        """Test index to coordinate conversion."""
        grid = GridType(shape=(5, 5), spacing=(0.1, 0.1), origin=(0.0, 0.0))
        
        coord = grid.index_to_coord((2, 3))
        assert abs(coord[0] - 0.2) < 1e-10
        assert abs(coord[1] - 0.3) < 1e-10
    
    def test_grid_coord_to_index(self):
        """Test coordinate to index conversion."""
        grid = GridType(shape=(10, 10), spacing=(0.1, 0.1), origin=(0.0, 0.0))
        
        idx = grid.coord_to_index((0.25, 0.35))
        assert idx == (2, 3)
    
    def test_staggered_grid(self):
        """Test staggered grid creation."""
        grid = GridType(shape=(10, 10), spacing=(0.1, 0.1))
        
        u_grid = grid.get_component_grid(0)
        v_grid = grid.get_component_grid(1)
        
        # Velocity grids have one fewer point in their direction
        assert u_grid.shape == (9, 10)
        assert v_grid.shape == (10, 9)


class TestStencilLibrary:
    """Test stencil generation and application."""
    
    def test_1d_central_2(self):
        """Test 2nd order central difference stencil."""
        stencil = stencil_1d_central_2()
        
        assert len(stencil.pattern) == 2
        assert stencil.accuracy == 2
        assert stencil.stencil_type == StencilType.CENTRAL
    
    def test_1d_forward_1(self):
        """Test 1st order forward difference stencil."""
        stencil = stencil_1d_forward_1()
        
        assert len(stencil.pattern) == 2
        assert stencil.accuracy == 1
        assert stencil.stencil_type == StencilType.FORWARD
    
    def test_1d_backward_1(self):
        """Test 1st order backward difference stencil."""
        stencil = stencil_1d_backward_1()
        
        assert len(stencil.pattern) == 2
        assert stencil.accuracy == 1
        assert stencil.stencil_type == StencilType.BACKWARD
    
    def test_1d_second_central_2(self):
        """Test 2nd order central 2nd derivative stencil."""
        stencil = stencil_1d_second_central_2()
        
        assert len(stencil.pattern) == 3
        assert stencil.accuracy == 2
        # Coefficients should be [1, -2, 1]
        np.testing.assert_array_almost_equal(
            stencil.coefficients, np.array([1.0, -2.0, 1.0])
        )
    
    def test_5_point_laplacian(self):
        """Test 5-point 2D Laplacian stencil."""
        stencil = stencil_5_point()
        
        assert stencil.accuracy == 2
        assert len(stencil.pattern) == 5
        # Center coefficient should be -4
        center_idx = stencil.pattern.index((0, 0))
        assert abs(stencil.coefficients[center_idx] + 4.0) < 1e-10
    
    def test_7_point_3d_laplacian(self):
        """Test 7-point 3D Laplacian stencil."""
        stencil = stencil_7_point_3d()
        
        assert stencil.accuracy == 2
        assert len(stencil.pattern) == 7
    
    def test_stencil_apply(self):
        """Test stencil application to field."""
        stencil = stencil_1d_second_central_2()
        field = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # At index 2: (1*5.0 - 2*3.0 + 1*1.0) = 0 (for uniform values)
        result = stencil.apply(field, (2,))
        assert abs(result) < 1e-10
    
    def test_stencil_apply_2d(self):
        """Test 2D stencil application."""
        stencil = stencil_5_point()
        field = np.ones((5, 5))
        
        result = stencil.apply(field, (2, 2))
        assert abs(result) < 1e-10
    
    def test_apply_stencil_field(self):
        """Test applying stencil to entire field."""
        stencil = stencil_1d_second_central_2()
        field = np.linspace(0, 1, 10)
        
        result = apply_stencil_field(stencil, field, field.shape)
        
        # Result should be small for linear function (second derivative = 0)
        assert np.all(np.abs(result[1:-1]) < 1e-10)
    
    def test_scale_stencil(self):
        """Test stencil scaling."""
        stencil = stencil_1d_central_2()
        scaled = scale_stencil(stencil, 2.0)
        
        np.testing.assert_array_almost_equal(
            scaled.coefficients, stencil.coefficients * 2.0
        )


class TestQuadratureRules:
    """Test quadrature rule implementation."""
    
    def test_gauss_legendre_1(self):
        """Test 1-point Gauss-Legendre quadrature."""
        quad = gauss_legendre_1()
        
        assert quad.degree == 1
        assert len(quad.nodes) == 1
        assert len(quad.weights) == 1
        # Integral of constant should give correct result
        result = integrate(quad, lambda x: 1.0)
        assert abs(result - 2.0) < 1e-10  # Interval [-1, 1]
    
    def test_gauss_legendre_2(self):
        """Test 2-point Gauss-Legendre quadrature."""
        quad = gauss_legendre_2()
        
        assert quad.degree == 3
        assert len(quad.nodes) == 2
        assert len(quad.weights) == 2
        # Weights should sum to 2
        assert abs(np.sum(quad.weights) - 2.0) < 1e-10
    
    def test_gauss_legendre_3(self):
        """Test 3-point Gauss-Legendre quadrature."""
        quad = gauss_legendre_3()
        
        assert quad.degree == 5
        assert len(quad.nodes) == 3
    
    def test_simpson_rule(self):
        """Test Simpson's rule."""
        quad = simpson_rule()
        
        assert quad.degree == 3
        assert len(quad.nodes) == 3
        # Weights should sum to 2
        assert abs(np.sum(quad.weights) - 2.0) < 1e-10
    
    def test_trapezoidal_rule(self):
        """Test trapezoidal rule."""
        quad = trapezoidal_rule(n_intervals=4)
        
        assert quad.degree == 1
        assert len(quad.nodes) == 5
    
    def test_triangle_quadrature_1(self):
        """Test 1-point triangle quadrature."""
        quad = quadrature_triangle_1()
        
        assert quad.degree == 1
        assert quad.domain == "triangle"
        assert len(quad.nodes) == 1
        # Weight should be 0.5 (area of reference triangle)
        assert abs(quad.weights[0] - 0.5) < 1e-10
    
    def test_triangle_quadrature_3(self):
        """Test 3-point triangle quadrature."""
        quad = quadrature_triangle_3()
        
        assert quad.degree == 2
        assert len(quad.nodes) == 3
        # Weights should sum to 0.5
        assert abs(np.sum(quad.weights) - 0.5) < 1e-10
    
    def test_tetrahedron_quadrature_1(self):
        """Test 1-point tetrahedron quadrature."""
        quad = quadrature_tetrahedron_1()
        
        assert quad.degree == 1
        assert quad.domain == "tetrahedron"
        assert len(quad.nodes) == 1
        # Weight should be 1/6 (volume of reference tet)
        assert abs(quad.weights[0] - 1.0/6.0) < 1e-10
    
    def test_2d_tensor_quadrature(self):
        """Test 2D tensor product quadrature."""
        quad = gauss_legendre_tensor_2d(n=2)
        
        assert quad.domain == "quadrilateral"
        # 2x2 points
        assert len(quad.nodes) == 4
        # Weights should sum to 4 (area of [-1,1]^2)
        assert abs(np.sum(quad.weights) - 4.0) < 1e-10
    
    def test_3d_tensor_quadrature(self):
        """Test 3D tensor product quadrature."""
        quad = gauss_legendre_tensor_3d(n=2)
        
        assert quad.domain == "hexahedron"
        # 2x2x2 points
        assert len(quad.nodes) == 8
        # Weights should sum to 8 (volume of [-1,1]^3)
        assert abs(np.sum(quad.weights) - 8.0) < 1e-10


class TestStabilityAnalysis:
    """Test stability analysis functions."""
    
    def test_cfl_condition_advection(self):
        """Test CFL condition for advection."""
        stencil = stencil_1d_central_2()
        dt = 0.1
        dx = 1.0
        max_velocity = 1.0
        
        is_stable, cfl = check_cfl_condition(stencil, dt, dx, max_velocity)
        
        # For central differencing, CFL should be computed
        assert isinstance(is_stable, bool)
        assert isinstance(cfl, float)
    
    def test_cfl_condition_diffusion(self):
        """Test CFL condition for diffusion."""
        stencil = stencil_1d_second_central_2()
        dt = 0.01
        dx = 1.0
        max_velocity = 1.0
        
        is_stable, cfl = check_cfl_condition(stencil, dt, dx, max_velocity)
        
        # Diffusion CFL = dt/dx^2, should be <= 0.5
        expected_cfl = dt / (dx ** 2)
        assert abs(cfl - expected_cfl) < 1e-10
    
    def test_spectral_radius_1d(self):
        """Test spectral radius computation for 1D stencil."""
        grid = GridType(shape=(10,), spacing=(1.0,))
        stencil = stencil_1d_second_central_2()
        
        rho = compute_spectral_radius(stencil, grid)
        
        assert rho >= 0
        assert isinstance(rho, float)
    
    def test_spectral_radius_2d(self):
        """Test spectral radius computation for 2D stencil."""
        grid = GridType(shape=(10, 10), spacing=(1.0, 1.0))
        stencil = stencil_5_point()
        
        rho = compute_spectral_radius(stencil, grid)
        
        assert rho >= 0
    
    def test_lax_equivalence(self):
        """Test Lax equivalence check."""
        stencil = stencil_1d_central_2()
        
        is_consistent, order = check_lax_equivalence(stencil)
        
        assert is_consistent == True
        assert order == stencil.accuracy
    
    def test_condition_number(self):
        """Test condition number estimation."""
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        
        cond = estimate_condition_number(A, method="exact")
        
        assert cond > 0
        # For this matrix, exact condition number is 3
        assert abs(cond - 3.0) < 1e-6
    
    def test_analyze_stability(self):
        """Test comprehensive stability analysis."""
        grid = GridType(shape=(10, 10), spacing=(0.1, 0.1))
        stencil = stencil_5_point()
        
        stability = analyze_stability(stencil, grid)
        
        assert isinstance(stability, StabilityInfo)
        assert stability.spectral_radius is not None
        assert isinstance(stability.is_stable, bool)
    
    def test_time_step_stability(self):
        """Test time step stability check."""
        grid = GridType(shape=(10,), spacing=(1.0,))
        stencil = stencil_1d_central_2()
        
        result = check_time_step_stability(stencil, grid, dt=0.5)
        
        assert result.is_stable is not None
        assert result.metric_name == "time_step"


class TestDiscreteLowering:
    """Test discrete lowering from CALC/GEO to DISC."""
    
    def test_lower_gradient_1d(self):
        """Test gradient lowering in 1D."""
        grid = GridType(shape=(10,), spacing=(0.1,))
        context = LoweringContext(grid=grid, quadrature=gauss_legendre_2())
        lowerer = DiscreteLowerer(context)
        
        stencils = lowerer.lower_gradient(None, grid)
        
        assert len(stencils) == 1
        assert stencils[0].stencil_type == StencilType.CENTRAL
    
    def test_lower_gradient_2d(self):
        """Test gradient lowering in 2D."""
        grid = GridType(shape=(10, 10), spacing=(0.1, 0.1))
        context = LoweringContext(grid=grid, quadrature=gauss_legendre_2())
        lowerer = DiscreteLowerer(context)
        
        stencils = lowerer.lower_gradient(None, grid)
        
        assert len(stencils) == 2
    
    def test_lower_gradient_3d(self):
        """Test gradient lowering in 3D."""
        grid = GridType(shape=(10, 10, 10), spacing=(0.1, 0.1, 0.1))
        context = LoweringContext(grid=grid, quadrature=gauss_legendre_3())
        lowerer = DiscreteLowerer(context)
        
        stencils = lowerer.lower_gradient(None, grid)
        
        assert len(stencils) == 3
    
    def test_lower_divergence(self):
        """Test divergence lowering."""
        grid = GridType(shape=(10, 10), spacing=(0.1, 0.1))
        context = LoweringContext(grid=grid, quadrature=gauss_legendre_2())
        lowerer = DiscreteLowerer(context)
        
        stencils = lowerer.lower_divergence(None, grid)
        
        assert len(stencils) == 2
    
    def test_lower_curl_2d(self):
        """Test curl lowering in 2D."""
        grid = GridType(shape=(10, 10), spacing=(0.1, 0.1))
        context = LoweringContext(grid=grid, quadrature=gauss_legendre_2())
        lowerer = DiscreteLowerer(context)
        
        stencils = lowerer.lower_curl(None, grid)
        
        assert len(stencils) == 1
    
    def test_lower_curl_3d(self):
        """Test curl lowering in 3D."""
        grid = GridType(shape=(10, 10, 10), spacing=(0.1, 0.1, 0.1))
        context = LoweringContext(grid=grid, quadrature=gauss_legendre_3())
        lowerer = DiscreteLowerer(context)
        
        stencils = lowerer.lower_curl(None, grid)
        
        assert len(stencils) == 3
    
    def test_lower_laplacian_standard(self):
        """Test Laplacian lowering with standard scheme."""
        grid = GridType(shape=(10, 10), spacing=(0.1, 0.1))
        context = LoweringContext(grid=grid, quadrature=gauss_legendre_2())
        lowerer = DiscreteLowerer(context)
        
        stencil = lowerer.lower_laplacian(None, grid, scheme="standard")
        
        assert stencil.accuracy == 2
    
    def test_lower_laplacian_5_point(self):
        """Test Laplacian lowering with 5-point scheme."""
        grid = GridType(shape=(10, 10), spacing=(0.1, 0.1))
        context = LoweringContext(grid=grid, quadrature=gauss_legendre_2())
        lowerer = DiscreteLowerer(context)
        
        stencil = lowerer.lower_laplacian(None, grid, scheme="5-point")
        
        assert stencil is not None
    
    def test_lower_time_derivative(self):
        """Test time derivative lowering."""
        context = LoweringContext(
            grid=GridType(shape=(10,), spacing=(0.1,)),
            quadrature=gauss_legendre_1()
        )
        lowerer = DiscreteLowerer(context)
        
        stencil = lowerer.lower_time_derivative(None, scheme="forward_euler")
        
        assert stencil.accuracy == 1
    
    def test_create_fem_space(self):
        """Test FEM space creation."""
        context = LoweringContext(
            grid=GridType(shape=(10,), spacing=(0.1,)),
            quadrature=gauss_legendre_3()
        )
        lowerer = DiscreteLowerer(context)
        
        space = lowerer.create_fem_space("lagrange", degree=2, dim=2)
        
        assert space.element_type == FEMElementType.LAGRANGE
        assert space.degree == 2
    
    def test_compute_mass_matrix(self):
        """Test mass matrix computation."""
        context = LoweringContext(
            grid=GridType(shape=(10,), spacing=(0.1,)),
            quadrature=gauss_legendre_2()
        )
        lowerer = DiscreteLowerer(context)
        
        space = FEMSpace(element_type=FEMElementType.LAGRANGE, degree=1, dim=10)
        M = lowerer.compute_mass_matrix(space)
        
        assert M.shape == (10, 10)
    
    def test_lower_to_disc(self):
        """Test full lowering to DISC."""
        grid = GridType(shape=(10, 10), spacing=(0.1, 0.1))
        context = LoweringContext(
            grid=grid,
            quadrature=gauss_legendre_2()
        )
        lowerer = DiscreteLowerer(context)
        
        disc_repr = lowerer.lower_to_disc(None)
        
        assert "operators" in disc_repr
        assert "gradient" in disc_repr["operators"]
        assert "laplacian" in disc_repr["operators"]
        assert disc_repr["stability_info"] is not None


class TestMultiDimensionalDiscretizations:
    """Test multi-dimensional discretization features."""
    
    def test_isotropic_laplacian_1d(self):
        """Test dimension-independent Laplacian in 1D."""
        stencil = stencil_isotropic_laplacian(dim=1, dx=0.1)
        
        assert stencil is not None
    
    def test_isotropic_laplacian_2d(self):
        """Test dimension-independent Laplacian in 2D."""
        stencil = stencil_isotropic_laplacian(dim=2, dx=0.1)
        
        assert stencil is not None
        # Should be scaled by 1/dx^2
        center_idx = stencil.pattern.index((0, 0))
        assert abs(stencil.coefficients[center_idx] + 400.0) < 1.0  # -4/0.01
    
    def test_isotropic_laplacian_3d(self):
        """Test dimension-independent Laplacian in 3D."""
        stencil = stencil_isotropic_laplacian(dim=3, dx=0.1)
        
        assert stencil is not None
    
    def test_gradient_nd(self):
        """Test N-dimensional gradient stencils."""
        for dim in [1, 2, 3]:
            stencils = stencil_gradient_nd(dim)
            assert len(stencils) == dim
    
    def test_divergence_nd(self):
        """Test N-dimensional divergence stencils."""
        for dim in [1, 2, 3]:
            stencils = stencil_divergence_nd(dim)
            assert len(stencils) == dim
    
    def test_staggered_grid_components(self):
        """Test staggered grid component grids."""
        base = GridType(shape=(10, 10), spacing=(0.1, 0.1))
        grids = create_staggered_grid(base, ["u", "v", "p"])
        
        assert "p" in grids
        assert "u" in grids
        assert "v" in grids
        # Velocity grids should be staggered
        assert grids["u"].shape[0] == 9  # Staggered in x
        assert grids["v"].shape[1] == 9  # Staggered in y


class TestDiscreteOperators:
    """Test discrete operator types."""
    
    def test_discrete_field_norm(self):
        """Test discrete field norm computation."""
        space = FDSpace(
            stencil=stencil_1d_central_2(),
            order=2,
            grid=GridType(shape=(10,), spacing=(0.1,))
        )
        field = DiscreteField(space=space, values=np.random.rand(10))
        
        norm = field.norm(p=2)
        
        assert norm >= 0
    
    def test_stability_info(self):
        """Test stability info creation."""
        stability = StabilityInfo(
            cond_number=1.5,
            spectral_radius=0.9,
            CFL_limit=0.5,
            is_stable=True
        )
        
        assert stability.cond_number == 1.5
        assert stability.spectral_radius == 0.9
        assert stability.CFL_limit == 0.5
        assert stability.is_stable == True


class TestBoundaryConditions:
    """Test boundary condition handling."""
    
    def test_boundary_indices(self):
        """Test boundary index generation."""
        grid = GridType(shape=(5, 5), spacing=(0.1, 0.1))
        
        x_min_indices = grid.boundary_indices("x_min")
        x_max_indices = grid.boundary_indices("x_max")
        
        assert len(x_min_indices) == 5
        assert len(x_max_indices) == 5
        # First index should be 0 for x_min
        assert all(idx[0] == 0 for idx in x_min_indices)
        # First index should be 4 for x_max
        assert all(idx[0] == 4 for idx in x_max_indices)
    
    def test_interior_indices(self):
        """Test interior index generation."""
        grid = GridType(shape=(5, 5), spacing=(0.1, 0.1))
        
        interior = grid.interior_indices()
        
        # Interior should exclude boundaries
        assert len(interior) == 9  # 3x3 interior
    
    def test_is_boundary(self):
        """Test boundary check."""
        grid = GridType(shape=(5, 5), spacing=(0.1, 0.1))
        
        assert grid.is_boundary((0, 2)) == True
        assert grid.is_boundary((2, 2)) == False
        assert grid.is_boundary((4, 4)) == True
    
    def test_apply_dirichlet_bc(self):
        """Test Dirichlet boundary condition application."""
        grid = GridType(shape=(5, 5), spacing=(0.1, 0.1))
        field = np.ones((5, 5))
        
        result = grid.apply_boundary_condition(
            field, BoundaryConditionType.DIRICHLET, bc_value=0.0, face="x_min"
        )
        
        assert np.all(result[0, :] == 0.0)
        # Other boundaries should be unchanged
        assert np.all(result[-1, :] == 1.0)


class TestIntegrationExamples:
    """Test complete integration examples."""
    
    def test_heat_equation_discretization(self):
        """Test heat equation discretization."""
        # Create grid
        grid = GridType(shape=(50, 50), spacing=(0.02, 0.02))
        
        # Create Laplacian stencil
        lap_stencil = stencil_isotropic_laplacian(dim=2, dx=0.02)
        
        # Analyze stability
        stability = analyze_stability(lap_stencil, grid)
        
        # For heat equation, diffusion number should be <= 0.25 for stability
        # (explicit Euler with 2nd order space)
        assert stability.is_stable or stability.CFL_limit is not None
    
    def test_wave_equation_discretization(self):
        """Test wave equation discretization."""
        # Create grid
        grid = GridType(shape=(100,), spacing=(0.01,))
        
        # Create 2nd derivative stencil
        d2_stencil = stencil_1d_second_central_2()
        
        # Check CFL condition for wave equation
        # CFL = c * dt / dx <= 1
        is_stable, cfl = check_cfl_condition(
            d2_stencil, dt=0.005, dx=0.01, max_velocity=1.0
        )
        
        # For explicit wave equation with central differencing,
        # CFL should be <= 1
        assert cfl == 0.5  # 0.005 / 0.01 = 0.5
    
    def test_poisson_problem_setup(self):
        """Test Poisson problem setup."""
        # Grid
        grid = GridType(shape=(20, 20), spacing=(0.05, 0.05))
        
        # Lower operators
        context = LoweringContext(
            grid=grid,
            quadrature=gauss_legendre_tensor_2d(n=2)
        )
        lowerer = DiscreteLowerer(context)
        
        # Get Laplacian
        lap_stencil = lowerer.lower_laplacian(None, grid, scheme="5-point")
        
        # Get mass matrix for RHS
        space = lowerer.create_fem_space("lagrange", degree=1, dim=2)
        mass_matrix = lowerer.compute_mass_matrix(space)
        
        # Verify shapes
        assert mass_matrix.shape[0] == space.dim


# === Utility Tests ===

class TestUtilities:
    """Test utility functions."""
    
    def test_create_uniform_grid(self):
        """Test uniform grid creation helper."""
        grid = create_uniform_grid(shape=(10, 10), spacing=0.1)
        
        assert grid.shape == (10, 10)
        assert grid.spacing == (0.1, 0.1)
    
    def test_create_uniform_grid_with_origin(self):
        """Test uniform grid with origin."""
        grid = create_uniform_grid(
            shape=(10, 10), spacing=0.1, origin=(1.0, 2.0)
        )
        
        assert grid.origin == (1.0, 2.0)
        # First point should be at origin
        coord = grid.index_to_coord((0, 0))
        assert abs(coord[0] - 1.0) < 1e-10
        assert abs(coord[1] - 2.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
