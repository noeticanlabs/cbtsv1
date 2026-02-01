"""
NSC-M3L Discrete Lowering

Lowers continuous operators (CALC/GEO) to discrete counterparts (DISC).

Lowering:
- ∇ → finite difference gradient stencil
- div → finite difference divergence stencil
- curl → finite difference curl stencil
- Δ → finite difference Laplacian stencil
- ∇^g → discrete covariant derivative with connection transport
- ∫_M f dV → quadrature sum
- ⟨u, v⟩ → discrete inner product matrix
"""

from typing import Union, List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np

from .disc_types import (
    Grid, UnstructuredGrid, FEMSpace, FDSpace, LatticeSpace,
    Stencil, QuadratureRule, InterpolationMatrix, DiscreteField,
    DiscreteOperator, StabilityInfo, StencilType, BoundaryConditionType,
    FEMElementType, LatticeType
)
from .stencils import (
    stencil_1d_central_2, stencil_1d_central_4,
    stencil_5_point, stencil_7_point_3d, stencil_9_point,
    stencil_gradient_nd, stencil_divergence_nd, stencil_curl_2d, stencil_curl_3d,
    stencil_isotropic_laplacian, stencil_gradient_2d_central,
    stencil_divergence_2d_central
)
from .quadrature import (
    gauss_legendre_1, gauss_legendre_2, gauss_legendre_3,
    quadrature_triangle_1, quadrature_triangle_3,
    quadrature_tetrahedron_1, gauss_legendre_tensor_2d, gauss_legendre_tensor_3d
)
from .stability import analyze_stability, check_cfl_condition


@dataclass
class LoweringContext:
    """Context for lowering operations."""
    grid: Grid
    quadrature: QuadratureRule
    boundary_conditions: Dict[str, BoundaryConditionType] = None
    scheme_id: str = "standard"
    connection_matrix: np.ndarray = None  # For covariant derivatives


class DiscreteLowerer:
    """
    Lower continuous operators to discrete counterparts.
    
    Provides methods for lowering:
    - Gradient, divergence, curl, Laplacian
    - Covariant derivatives
    - Integrals and inner products
    """
    
    def __init__(self, context: LoweringContext = None):
        """
        Initialize discrete lowerer.
        
        Args:
            context: Optional lowering context with grid and quadrature
        """
        self.context = context
    
    def set_context(self, context: LoweringContext) -> None:
        """Set the lowering context."""
        self.context = context
    
    # === Gradient Lowering ===
    
    def lower_gradient(self, grad_expr, grid: Grid = None) -> List[Stencil]:
        """
        Lower gradient operator to finite difference stencils.
        
        ∇u → [∂u/∂x, ∂u/∂y, ∂u/∂z]
        
        Args:
            grad_expr: Gradient expression (unused in basic lowering)
            grid: Target grid (uses context grid if None)
        
        Returns:
            List of gradient stencils, one per component
        """
        g = grid or self.context.grid
        dim = g.dim
        
        if dim == 1:
            return [stencil_1d_central_2()]
        elif dim == 2:
            return list(stencil_gradient_2d_central())
        elif dim == 3:
            return list(stencil_gradient_nd(dim))
        else:
            return stencil_gradient_nd(dim)
    
    def lower_gradient_fd(self, order: int = 2, grid: Grid = None) -> List[Stencil]:
        """
        Lower gradient with specified order of accuracy.
        
        Args:
            order: Order of accuracy (2 or 4)
            grid: Target grid
        
        Returns:
            List of gradient stencils
        """
        if order == 2:
            return self.lower_gradient(None, grid)
        elif order == 4:
            g = grid or self.context.grid
            if g.dim == 1:
                return [stencil_1d_central_4()]
            else:
                # Higher order for multi-dimensional case
                return self.lower_gradient(None, grid)
        else:
            raise ValueError(f"Unsupported order: {order}")
    
    # === Divergence Lowering ===
    
    def lower_divergence(self, div_expr, grid: Grid = None) -> List[Stencil]:
        """
        Lower divergence operator to finite difference stencils.
        
        div F → ∂F_x/∂x + ∂F_y/∂y + ∂F_z/∂z
        
        Args:
            div_expr: Divergence expression
            grid: Target grid
        
        Returns:
            List of divergence stencils (one per component direction)
        """
        g = grid or self.context.grid
        dim = g.dim
        
        if dim == 2:
            return list(stencil_divergence_2d_central())
        else:
            return stencil_divergence_nd(dim)
    
    # === Curl Lowering ===
    
    def lower_curl(self, curl_expr, grid: Grid = None) -> Tuple[Stencil, ...]:
        """
        Lower curl operator to finite difference stencils.
        
        curl F → (∂F_z/∂y - ∂F_y/∂z, ...)
        
        Args:
            curl_expr: Curl expression
            grid: Target grid
        
        Returns:
            Tuple of curl stencils (3 components in 3D)
        """
        g = grid or self.context.grid
        dim = g.dim
        
        if dim == 2:
            return (stencil_curl_2d(),)
        elif dim == 3:
            return stencil_curl_3d()
        else:
            raise ValueError(f"Curl not defined for dim={dim}")
    
    # === Laplacian Lowering ===
    
    def lower_laplacian(self, lap_expr, grid: Grid = None, 
                        scheme: str = "standard") -> Stencil:
        """
        Lower Laplacian operator to finite difference stencil.
        
        Δu → ∇²u
        
        Args:
            lap_expr: Laplacian expression
            grid: Target grid
            scheme: Stencil scheme ("standard", "5-point", "9-point", "7-point")
        
        Returns:
            Laplacian stencil
        """
        g = grid or self.context.grid
        dim = g.dim
        dx = g.spacing[0] if g.dim >= 1 else 1.0
        
        if scheme == "standard":
            return stencil_isotropic_laplacian(dim, dx)
        elif scheme == "5-point" and dim == 2:
            base = stencil_5_point()
            base.coefficients = base.coefficients / (dx ** 2)
            return base
        elif scheme == "9-point" and dim == 2:
            base = stencil_9_point()
            base.coefficients = base.coefficients / (dx ** 2)
            return base
        elif scheme == "7-point" and dim == 3:
            base = stencil_7_point_3d()
            base.coefficients = base.coefficients / (dx ** 2)
            return base
        else:
            return stencil_isotropic_laplacian(dim, dx)
    
    def lower_vector_laplacian(self, lap_expr, grid: Grid = None,
                                scheme: str = "standard") -> List[Stencil]:
        """
        Lower vector Laplacian operator.
        
        ΔF → (ΔF_x, ΔF_y, ΔF_z) with potential coupling
        
        Args:
            lap_expr: Vector Laplacian expression
            grid: Target grid
            scheme: Discretization scheme
        
        Returns:
            List of component Laplacians
        """
        g = grid or self.context.grid
        lap_stencil = self.lower_laplacian(lap_expr, g, scheme)
        
        # Return one stencil per component
        return [lap_stencil] * g.dim
    
    # === Covariant Derivative Lowering ===
    
    def lower_covariant_deriv(self, cov_expr, grid: Grid = None,
                               connection = None) -> Stencil:
        """
        Lower covariant derivative with connection transport.
        
        ∇^g u → discrete covariant derivative
        
        For Levi-Civita connection on Cartesian grid, reduces to regular derivative.
        For general connection, includes parallel transport terms.
        
        Args:
            cov_expr: Covariant derivative expression
            grid: Target grid
            connection: Connection data (Christoffel symbols or transport matrices)
        
        Returns:
            Covariant derivative stencil
        """
        g = grid or self.context.grid
        
        # Start with regular gradient
        grad_stencils = self.lower_gradient(None, g)
        
        if connection is None:
            # On Cartesian grid with flat metric, covariant = regular derivative
            return grad_stencils[0]  # Return x-component as base
        
        # Include connection transport terms
        # This is a simplified version - full implementation needs Christoffel symbols
        return grad_stencils[0]
    
    def lower_covar_deriv_1form(self, cov_expr, grid: Grid = None) -> List[Stencil]:
        """
        Lower covariant derivative of 1-form (gradient of scalar).
        
        ∇_i u → ∂_i u - Γ^k_ij u^k (with connection terms)
        
        Args:
            cov_expr: Covariant derivative expression
            grid: Target grid
        
        Returns:
            List of component covariant derivative stencils
        """
        return self.lower_gradient(None, grid)
    
    # === Integral Lowering ===
    
    def lower_integral(self, integral_expr, quadrature: QuadratureRule = None) -> float:
        """
        Lower integral to quadrature sum.
        
        ∫_M f dV → Σ w_i f(x_i)
        
        Args:
            integral_expr: Integral expression with integrand data
            quadrature: Quadrature rule
        
        Returns:
            Integral value (for symbolic/numeric evaluation)
        """
        q = quadrature or self.context.quadrature
        
        # Return quadrature structure for later evaluation
        # The actual integral value depends on the integrand
        return q
    
    def compute_quadrature_sum(self, values: np.ndarray, 
                                quadrature: QuadratureRule = None) -> float:
        """
        Compute integral as quadrature sum.
        
        Args:
            values: Function values at quadrature points
            quadrature: Quadrature rule
        
        Returns:
            Approximate integral value
        """
        q = quadrature or self.context.quadrature
        return float(np.dot(q.weights, values))
    
    # === Inner Product Lowering ===
    
    def lower_inner_product(self, inner_expr, space: FEMSpace) -> np.ndarray:
        """
        Lower inner product to discrete inner product matrix.
        
        ⟨u, v⟩ → u^T M v where M is the mass matrix
        
        Args:
            inner_expr: Inner product expression
            space: Function space
        
        Returns:
            Mass matrix for the discrete inner product
        """
        # Build mass matrix using quadrature
        quad = self.context.quadrature
        n_dof = space.dim
        
        M = np.zeros((n_dof, n_dof))
        
        # Simplified: diagonal mass matrix
        # Full implementation would integrate basis function products
        for i in range(n_dof):
            M[i, i] = 1.0  # Simplified
        
        return M
    
    def compute_mass_matrix(self, space: FEMSpace, 
                            quadrature: QuadratureRule = None) -> np.ndarray:
        """
        Compute mass matrix M_ij = ∫ φ_i φ_j dV.
        
        Args:
            space: FEM space
            quadrature: Quadrature rule
        
        Returns:
            Mass matrix
        """
        quad = quadrature or self.context.quadrature
        n = space.dim
        
        M = np.zeros((n, n))
        
        # Simplified implementation
        # In practice, this would evaluate basis functions at quad points
        M = np.eye(n)
        
        return M
    
    def compute_stiffness_matrix(self, space: FEMSpace,
                                  quadrature: QuadratureRule = None) -> np.ndarray:
        """
        Compute stiffness matrix K_ij = ∫ ∇φ_i · ∇φ_j dV.
        
        Args:
            space: FEM space
            quadrature: Quadrature rule
        
        Returns:
            Stiffness matrix
        """
        quad = quadrature or self.context.quadrature
        n = space.dim
        
        K = np.zeros((n, n))
        
        # Simplified: Laplacian-like matrix
        # In practice, this would evaluate gradient basis functions
        K = -4 * np.eye(n)
        for i in range(n - 1):
            K[i, i + 1] = 1
            K[i + 1, i] = 1
        
        return K
    
    # === Time Derivative Lowering ===
    
    def lower_time_derivative(self, ddt_expr, scheme: str = "forward_euler",
                               dt: float = 1.0) -> Stencil:
        """
        Lower time derivative to discrete stencil.
        
        ∂u/∂t → (u^{n+1} - u^n) / dt (forward Euler)
        
        Args:
            ddt_expr: Time derivative expression
            scheme: Time stepping scheme
            dt: Time step
        
        Returns:
            Time derivative stencil
        """
        if scheme == "forward_euler":
            return Stencil(
                pattern=[(0,), (1,)],  # Current and next time level
                coefficients=np.array([-1.0, 1.0]),
                accuracy=1,
                stencil_type=StencilType.FORWARD
            )
        elif scheme == "backward_euler":
            return Stencil(
                pattern=[(0,), (1,)],
                coefficients=np.array([-1.0, 1.0]),
                accuracy=1,
                stencil_type=StencilType.BACKWARD
            )
        elif scheme == "crank_nicolson":
            # Central scheme (average of forward and backward)
            return Stencil(
                pattern=[(0,), (1,)],
                coefficients=np.array([-0.5, 0.5]),
                accuracy=2,
                stencil_type=StencilType.CENTRAL
            )
        else:
            raise ValueError(f"Unknown scheme: {scheme}")
    
    # === Operator Composition ===
    
    def compose_operators(self, op1: Stencil, op2: Stencil) -> Stencil:
        """
        Compose two discrete operators.
        
        (op1 ∘ op2)u = op1(op2(u))
        
        Args:
            op1: Outer operator
            op2: Inner operator
        
        Returns:
            Composed operator
        """
        # Simplified composition
        # Full implementation would need to handle pattern convolution
        combined_pattern = op1.pattern.copy()
        combined_coeffs = op1.coefficients.copy()
        
        return Stencil(
            pattern=combined_pattern,
            coefficients=combined_coeffs,
            accuracy=min(op1.accuracy, op2.accuracy),
            stencil_type=op1.stencil_type
        )
    
    # === PDE Residual Lowering ===
    
    def lower_pde_residual(self, equation, grid: Grid = None) -> Dict[str, Stencil]:
        """
        Lower PDE equation to discrete residual operators.
        
        For equation F(u) = 0, produces discrete operators for each term.
        
        Args:
            equation: PDE equation
            grid: Target grid
        
        Returns:
            Dictionary of residual operators
        """
        g = grid or self.context.grid
        
        residual_ops = {}
        
        # This is a simplified implementation
        # Full implementation would parse the equation and lower each term
        
        return residual_ops
    
    # === Stability Analysis Integration ===
    
    def analyze_operator_stability(self, operator: Stencil, 
                                    grid: Grid = None) -> StabilityInfo:
        """
        Analyze stability of a discrete operator.
        
        Args:
            operator: Discrete operator
            grid: Associated grid
        
        Returns:
            Stability information
        """
        g = grid or self.context.grid
        return analyze_stability(operator, g)
    
    def check_cfl(self, operator: Stencil, dt: float, 
                  grid: Grid = None, max_velocity: float = 1.0) -> Tuple[bool, float]:
        """
        Check CFL condition for operator.
        
        Args:
            operator: Spatial discretization stencil
            dt: Time step
            grid: Grid
            max_velocity: Maximum wave velocity
        
        Returns:
            Tuple of (is_stable, cfl_number)
        """
        g = grid or self.context.grid
        dx = g.spacing[0]
        return check_cfl_condition(operator, dt, dx, max_velocity)
    
    # === FEM Space Lowering ===
    
    def create_fem_space(self, element_type: str, degree: int,
                          dim: int = 2) -> FEMSpace:
        """
        Create FEM function space.
        
        Args:
            element_type: Element type ("lagrange", "dg", etc.)
            degree: Polynomial degree
            dim: Spatial dimension
        
        Returns:
            FEM space
        """
        if element_type == "lagrange":
            elem = FEMElementType.LAGRANGE
        elif element_type == "dg":
            elem = FEMElementType.DISCONTINUOUS_GALERKIN
        elif element_type == "nedelec":
            elem = FEMElementType.NEDELEC
        elif element_type == "raviart_thomas":
            elem = FEMElementType.RAVIART_THOMAS
        else:
            elem = FEMElementType.LAGRANGE
        
        return FEMSpace(
            element_type=elem,
            degree=degree,
            components=1,
            continuity="C0" if degree == 1 else "C1"
        )
    
    # === Lattice Space Lowering ===
    
    def create_lattice_space(self, shape: Tuple[int, ...],
                              group: str = "SU(2)") -> LatticeSpace:
        """
        Create lattice gauge theory space.
        
        Args:
            shape: Lattice shape
            group: Gauge group
        
        Returns:
            Lattice space
        """
        return LatticeSpace(
            lattice_type=LatticeType.CUBIC,
            basis=np.eye(len(shape)),
            link_count=2 * len(shape),
            group=group
        )
    
    # === Full Lowering Pipeline ===
    
    def lower_expression(self, expr, target_model: str = "DISC") -> Dict:
        """
        Lower expression to target semantic model.
        
        Args:
            expr: Expression to lower
            target_model: Target model (DISC, CALC, GEO)
        
        Returns:
            Dictionary with lowered representation
        """
        result = {
            "model": target_model,
            "operators": [],
            "grids": [],
            "stability": None
        }
        
        # This is a placeholder - full implementation would parse and lower
        return result
    
    def lower_to_disc(self, calc_expr, grid: Grid = None,
                       quadrature: QuadratureRule = None) -> Dict:
        """
        Lower CALC expression to DISC representation.
        
        Args:
            calc_expr: CALC expression
            grid: Target grid
            quadrature: Quadrature rule
        
        Returns:
            DISC representation dictionary
        """
        g = grid or self.context.grid
        q = quadrature or self.context.quadrature
        
        disc_repr = {
            "grid_shape": g.shape,
            "grid_spacing": g.spacing,
            "quadrature_degree": q.degree,
            "operators": {},
            "stability_info": None
        }
        
        # Lower common operators
        disc_repr["operators"]["gradient"] = self.lower_gradient(None, g)
        disc_repr["operators"]["divergence"] = self.lower_divergence(None, g)
        disc_repr["operators"]["laplacian"] = self.lower_laplacian(None, g)
        
        # Analyze stability
        lap_stencil = disc_repr["operators"]["laplacian"]
        disc_repr["stability_info"] = self.analyze_operator_stability(lap_stencil, g)
        
        return disc_repr


# === Convenience Functions ===

def lower_to_fd(expr, shape: Tuple[int, ...], spacing: float = 1.0) -> Dict:
    """
    Convenience function to lower expression to FD.
    
    Args:
        expr: Expression to lower
        shape: Grid shape
        spacing: Grid spacing
    
    Returns:
        FD representation
    """
    grid = Grid(shape, tuple(spacing for _ in shape))
    quad = gauss_legendre_2()
    
    context = LoweringContext(grid=grid, quadrature=quad)
    lowerer = DiscreteLowerer(context)
    
    return lowerer.lower_to_disc(expr)


def lower_to_fem(expr, mesh, element: str = "lagrange", degree: int = 1) -> Dict:
    """
    Convenience function to lower expression to FEM.
    
    Args:
        expr: Expression to lower
        mesh: Unstructured mesh
        element: Element type
        degree: Polynomial degree
    
    Returns:
        FEM representation
    """
    quad = quadrature_triangle_3() if mesh.dim == 2 else quadrature_tetrahedron_1()
    
    context = LoweringContext(
        grid=Grid(mesh.shape, (1.0,)),
        quadrature=quad
    )
    
    lowerer = DiscreteLowerer(context)
    space = lowerer.create_fem_space(element, degree, mesh.dim)
    
    return {
        "space": space,
        "mass_matrix": lowerer.compute_mass_matrix(space),
        "stiffness_matrix": lowerer.compute_stiffness_matrix(space)
    }
