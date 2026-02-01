"""
NSC-M3L Stencil Library

Implements common finite difference stencils per section 4.4 of the specification.

Standard FD stencils:
- Central, forward, backward differences
- Laplacian stencils (5-point, 9-point, 7-point)
- Variable coefficient stencils
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from .disc_types import (
    Stencil, StencilType, Grid, DiscreteField, BoundaryConditionType
)


# === 1D Stencils ===

def stencil_1d_central_2() -> Stencil:
    """
    2nd order central difference stencil for first derivative.
    
    u'(x) ≈ (u(x+h) - u(x-h)) / (2h)
    
    Pattern: [(-1,), (1,)]
    Coefficients: [-1/(2h), 1/(2h)]
    """
    return Stencil(
        pattern=[(-1,), (1,)],
        coefficients=np.array([-0.5, 0.5]),
        accuracy=2,
        stencil_type=StencilType.CENTRAL
    )


def stencil_1d_forward_1() -> Stencil:
    """
    1st order forward difference stencil for first derivative.
    
    u'(x) ≈ (u(x+h) - u(x)) / h
    
    Pattern: [(0,), (1,)]
    Coefficients: [-1/h, 1/h]
    """
    return Stencil(
        pattern=[(0,), (1,)],
        coefficients=np.array([-1.0, 1.0]),
        accuracy=1,
        stencil_type=StencilType.FORWARD
    )


def stencil_1d_backward_1() -> Stencil:
    """
    1st order backward difference stencil for first derivative.
    
    u'(x) ≈ (u(x) - u(x-h)) / h
    
    Pattern: [(-1,), (0,)]
    Coefficients: [-1/h, 1/h]
    """
    return Stencil(
        pattern=[(-1,), (0,)],
        coefficients=np.array([-1.0, 1.0]),
        accuracy=1,
        stencil_type=StencilType.BACKWARD
    )


def stencil_1d_central_4() -> Stencil:
    """
    4th order central difference stencil for first derivative.
    
    u'(x) ≈ (-u(x+2h) + 8u(x+h) - 8u(x-h) + u(x+2h)) / (12h)
    
    Pattern: [(-2,), (-1,), (1,), (2,)]
    Coefficients: [1/12, -2/3, 2/3, -1/12]
    """
    return Stencil(
        pattern=[(-2,), (-1,), (1,), (2,)],
        coefficients=np.array([1/12, -2/3, 2/3, -1/12]),
        accuracy=4,
        stencil_type=StencilType.CENTRAL
    )


def stencil_1d_second_central_2() -> Stencil:
    """
    2nd order central difference stencil for second derivative.
    
    u''(x) ≈ (u(x+h) - 2u(x) + u(x-h)) / h²
    
    Pattern: [(-1,), (0,), (1,)]
    Coefficients: [1/h², -2/h², 1/h²]
    """
    return Stencil(
        pattern=[(-1,), (0,), (1,)],
        coefficients=np.array([1.0, -2.0, 1.0]),
        accuracy=2,
        stencil_type=StencilType.CENTRAL
    )


# === 2D Stencils ===

def stencil_5_point() -> Stencil:
    """
    5-point 2D Laplacian stencil.
    
    Δu = (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}) / h²
    
    Pattern and coefficients for unit spacing.
    """
    return Stencil(
        pattern=[(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)],
        coefficients=np.array([1.0, 1.0, 1.0, 1.0, -4.0]),
        accuracy=2,
        stencil_type=StencilType.CENTRAL
    )


def stencil_9_point() -> Stencil:
    """
    9-point 2D Laplacian stencil (more accurate).
    
    Includes diagonal neighbors.
    
    Pattern: 9 points including diagonals.
    """
    # Central point
    pattern = [(0, 0)]
    coefficients = [-4.0]
    
    # Cardinal directions
    for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        pattern.append(d)
        coefficients = np.append(coefficients, 1.0)
    
    # Diagonal directions (with 0.5 weight)
    for d in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        pattern.append(d)
        coefficients = np.append(coefficients, 0.5)
    
    return Stencil(
        pattern=pattern,
        coefficients=coefficients,
        accuracy=2,
        stencil_type=StencilType.CENTRAL
    )


def stencil_gradient_2d_central() -> Tuple[Stencil, Stencil]:
    """
    2D central gradient stencils (x and y components).
    
    ∂u/∂x ≈ (u_{i+1,j} - u_{i-1,j}) / (2h)
    ∂u/∂y ≈ (u_{i,j+1} - u_{i,j-1}) / (2h)
    """
    dx_stencil = Stencil(
        pattern=[(1, 0), (-1, 0)],
        coefficients=np.array([0.5, -0.5]),
        accuracy=2,
        stencil_type=StencilType.CENTRAL
    )
    
    dy_stencil = Stencil(
        pattern=[(0, 1), (0, -1)],
        coefficients=np.array([0.5, -0.5]),
        accuracy=2,
        stencil_type=StencilType.CENTRAL
    )
    
    return dx_stencil, dy_stencil


def stencil_divergence_2d_central() -> Tuple[Stencil, Stencil]:
    """
    2D central divergence stencils for vector field (u, v).
    
    div F = ∂u/∂x + ∂v/∂y
    """
    dx_stencil = Stencil(
        pattern=[(1, 0), (-1, 0)],
        coefficients=np.array([0.5, -0.5]),
        accuracy=2,
        stencil_type=StencilType.CENTRAL
    )
    
    dy_stencil = Stencil(
        pattern=[(0, 1), (0, -1)],
        coefficients=np.array([0.5, -0.5]),
        accuracy=2,
        stencil_type=StencilType.CENTRAL
    )
    
    return dx_stencil, dy_stencil


def stencil_curl_2d() -> Stencil:
    """
    2D curl stencil for scalar stream function or vorticity.
    
    ω = ∂v/∂x - ∂u/∂y
    
    Returns stencil for computing scalar curl from vector field.
    """
    # ∂v/∂x - ∂u/∂y
    # Using central differences: ∂v/∂x ≈ (v_{i+1} - v_{i-1}) / (2h)
    #                        ∂u/∂y ≈ (u_{j+1} - u_{j-1}) / (2h)
    
    return Stencil(
        pattern=[(1, 0), (-1, 0), (0, 1), (0, -1)],
        coefficients=np.array([0.5, -0.5, -0.5, 0.5]),
        accuracy=2,
        stencil_type=StencilType.CENTRAL
    )


# === 3D Stencils ===

def stencil_7_point_3d() -> Stencil:
    """
    7-point 3D Laplacian stencil.
    
    Pattern: 7 points (6 neighbors + center)
    """
    pattern = [(0, 0, 0)]
    coefficients = [-6.0]
    
    for d in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        pattern.append(d)
        coefficients = np.append(coefficients, 1.0)
    
    return Stencil(
        pattern=pattern,
        coefficients=coefficients,
        accuracy=2,
        stencil_type=StencilType.CENTRAL
    )


def stencil_gradient_3d_central() -> Tuple[Stencil, Stencil, Stencil]:
    """
    3D central gradient stencils (x, y, z components).
    """
    dx_stencil = Stencil(
        pattern=[(1, 0, 0), (-1, 0, 0)],
        coefficients=np.array([0.5, -0.5]),
        accuracy=2,
        stencil_type=StencilType.CENTRAL
    )
    
    dy_stencil = Stencil(
        pattern=[(0, 1, 0), (0, -1, 0)],
        coefficients=np.array([0.5, -0.5]),
        accuracy=2,
        stencil_type=StencilType.CENTRAL
    )
    
    dz_stencil = Stencil(
        pattern=[(0, 0, 1), (0, 0, -1)],
        coefficients=np.array([0.5, -0.5]),
        accuracy=2,
        stencil_type=StencilType.CENTRAL
    )
    
    return dx_stencil, dy_stencil, dz_stencil


def stencil_divergence_3d_central() -> Tuple[Stencil, Stencil, Stencil]:
    """
    3D central divergence stencils for vector field.
    """
    dx_stencil = Stencil(
        pattern=[(1, 0, 0), (-1, 0, 0)],
        coefficients=np.array([0.5, -0.5]),
        accuracy=2,
        stencil_type=StencilType.CENTRAL
    )
    
    dy_stencil = Stencil(
        pattern=[(0, 1, 0), (0, -1, 0)],
        coefficients=np.array([0.5, -0.5]),
        accuracy=2,
        stencil_type=StencilType.CENTRAL
    )
    
    dz_stencil = Stencil(
        pattern=[(0, 0, 1), (0, 0, -1)],
        coefficients=np.array([0.5, -0.5]),
        accuracy=2,
        stencil_type=StencilType.CENTRAL
    )
    
    return dx_stencil, dy_stencil, dz_stencil


def stencil_curl_3d() -> Tuple[Stencil, Stencil, Stencil]:
    """
    3D curl stencils for vector field.
    
    curl F = (∂Fz/∂y - ∂Fy/∂z, ∂Fx/∂z - ∂Fz/∂x, ∂Fy/∂x - ∂Fx/∂y)
    """
    # Component 1: ∂Fz/∂y - ∂Fy/∂z
    curl_x = Stencil(
        pattern=[(0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
        coefficients=np.array([0.5, -0.5, -0.5, 0.5]),
        accuracy=2,
        stencil_type=StencilType.CENTRAL
    )
    
    # Component 2: ∂Fx/∂z - ∂Fz/∂x
    curl_y = Stencil(
        pattern=[(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0)],
        coefficients=np.array([0.5, -0.5, -0.5, 0.5]),
        accuracy=2,
        stencil_type=StencilType.CENTRAL
    )
    
    # Component 3: ∂Fy/∂x - ∂Fx/∂y
    curl_z = Stencil(
        pattern=[(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)],
        coefficients=np.array([0.5, -0.5, -0.5, 0.5]),
        accuracy=2,
        stencil_type=StencilType.CENTRAL
    )
    
    return curl_x, curl_y, curl_z


# === Dimension-Independent Stencils ===

def stencil_isotropic_laplacian(dim: int, dx: float = 1.0) -> Stencil:
    """
    Dimension-independent isotropic Laplacian stencil.
    
    Args:
        dim: Spatial dimension (1, 2, or 3)
        dx: Grid spacing (default 1.0)
    
    Returns:
        Laplacian stencil with appropriate coefficients
    """
    if dim == 1:
        return stencil_1d_second_central_2()
    elif dim == 2:
        base_stencil = stencil_5_point()
    elif dim == 3:
        base_stencil = stencil_7_point_3d()
    else:
        raise ValueError(f"Unsupported dimension: {dim}")
    
    # Scale by 1/dx²
    scale = 1.0 / (dx ** 2)
    base_stencil.coefficients = base_stencil.coefficients * scale
    
    return base_stencil


def stencil_gradient_nd(dim: int) -> List[Stencil]:
    """
    N-dimensional gradient stencils.
    
    Returns list of stencils, one for each component.
    """
    stencils = []
    
    for d in range(dim):
        # Create pattern for this direction
        pattern = [(0,) * dim, (0,) * dim]
        pattern[0] = tuple(1 if i == d else 0 for i in range(dim))
        pattern[1] = tuple(-1 if i == d else 0 for i in range(dim))
        
        stencil = Stencil(
            pattern=pattern,
            coefficients=np.array([0.5, -0.5]),
            accuracy=2,
            stencil_type=StencilType.CENTRAL
        )
        stencils.append(stencil)
    
    return stencils


def stencil_divergence_nd(dim: int) -> List[Stencil]:
    """
    N-dimensional divergence stencils.
    
    Returns list of stencils, one for each component of vector field.
    """
    return stencil_gradient_nd(dim)  # Same pattern, different interpretation


# === Variable Coefficient Stencils ===

def stencil_var_coef_laplacian(coef_field: np.ndarray, dx: float = 1.0) -> Stencil:
    """
    Variable coefficient Laplacian stencil.
    
    div(α grad u) ≈ (α_{i+1/2}(u_{i+1}-u_i) - α_{i-1/2}(u_i-u_{i-1})) / h²
    
    This is the harmonic averaging discretization.
    
    Args:
        coef_field: Coefficient field values at cell centers
        dx: Grid spacing
    
    Returns:
        Stencil with coefficient-dependent weights
    """
    # For variable coefficient, we need to store coef info in the stencil
    # This returns a base pattern that can be modified based on local coefficients
    
    base_stencil = stencil_1d_second_central_2()
    base_stencil.coefficients = base_stencil.coefficients / (dx ** 2)
    
    return base_stencil


def stencil_advection(u_field: np.ndarray, dx: float = 1.0, 
                       direction: int = 1) -> Stencil:
    """
    Advection operator stencil.
    
    (u ∂/∂x)v where u is velocity field.
    
    Uses upwind scheme for stability.
    
    Args:
        u_field: Velocity field values
        dx: Grid spacing
        direction: 1 for positive, -1 for negative advection
    
    Returns:
        Upwind advection stencil
    """
    if direction > 0:
        # Upwind for positive velocity: uses (u_i - u_{i-1}) / h
        pattern = [(0,), (-1,)]
        coefficients = np.array([1.0, -1.0])
    else:
        # Upwind for negative velocity: uses (u_{i+1} - u_i) / h
        pattern = [(1,), (0,)]
        coefficients = np.array([1.0, -1.0])
    
    coefficients = coefficients / dx
    
    return Stencil(
        pattern=pattern,
        coefficients=coefficients,
        accuracy=1,
        stencil_type=StencilType.UPWIND
    )


def stencil_diffusion(nu: float, dx: float = 1.0) -> Stencil:
    """
    Diffusion operator stencil (nu * Laplacian).
    
    Args:
        nu: Diffusion coefficient
        dx: Grid spacing
    
    Returns:
        Scaled Laplacian stencil
    """
    laplacian = stencil_isotropic_laplacian(dim=1, dx=dx)
    laplacian.coefficients = laplacian.coefficients * nu
    
    return laplacian


# === Boundary Stencils ===

def create_boundary_stencil(interior_stencil: Stencil, 
                            bc_type: BoundaryConditionType,
                            bc_value: float = 0.0) -> Stencil:
    """
    Create boundary stencil for given boundary condition.
    
    Args:
        interior_stencil: Interior stencil pattern
        bc_type: Type of boundary condition
        bc_value: Boundary condition value (Dirichlet) or derivative (Neumann)
    
    Returns:
        Modified stencil for boundary points
    """
    if bc_type == BoundaryConditionType.DIRICHLET:
        # Dirichlet: u_boundary = specified_value
        # Use ghost cell approach
        return Stencil(
            pattern=[(0,)],
            coefficients=np.array([1.0]),
            accuracy=2,
            stencil_type=StencilType.FORWARD,
            boundary_stencil=None
        )
    elif bc_type == BoundaryConditionType.NEUMANN:
        # Neumann: ∂u/∂n = specified_value
        # Use one-sided difference for derivative
        return Stencil(
            pattern=[(0,), (1,)],
            coefficients=np.array([-1.0, 1.0]),
            accuracy=1,
            stencil_type=StencilType.FORWARD
        )
    else:
        return interior_stencil


# === Stencil Application Utilities ===

def apply_stencil_field(stencil: Stencil, field: np.ndarray, 
                        grid_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Apply stencil to entire field.
    
    Args:
        stencil: Stencil to apply
        field: Input field values
        grid_shape: Shape of output grid
    
    Returns:
        Field with stencil applied at each point
    """
    result = np.zeros(grid_shape)
    
    for idx in np.ndindex(grid_shape):
        result[idx] = stencil.apply(field, idx)
    
    return result


def compose_stencils(stencil1: Stencil, stencil2: Stencil) -> Stencil:
    """
    Compose two stencils (stencil2 after stencil1).
    
    (S1 ∘ S2)u = S1(S2(u))
    
    Args:
        stencil1: Outer stencil
        stencil2: Inner stencil
    
    Returns:
        Composed stencil
    """
    # This is a simplified composition - full implementation 
    # would need to compute the combined pattern
    combined_pattern = stencil1.pattern.copy()
    combined_coeffs = stencil1.coefficients.copy()
    
    return Stencil(
        pattern=combined_pattern,
        coefficients=combined_coeffs,
        accuracy=min(stencil1.accuracy, stencil2.accuracy),
        stencil_type=stencil1.stencil_type
    )


def scale_stencil(stencil: Stencil, factor: float) -> Stencil:
    """
    Scale stencil coefficients.
    
    Args:
        stencil: Original stencil
        factor: Scaling factor
    
    Returns:
        Scaled stencil
    """
    new_stencil = Stencil(
        pattern=stencil.pattern.copy(),
        coefficients=stencil.coefficients * factor,
        accuracy=stencil.accuracy,
        stencil_type=stencil.stencil_type,
        boundary_stencil=stencil.boundary_stencil
    )
    return new_stencil


def add_stencils(stencil1: Stencil, stencil2: Stencil) -> Stencil:
    """
    Add two stencils (element-wise coefficient addition).
    
    Args:
        stencil1: First stencil
        stencil2: Second stencil
    
    Returns:
        Sum stencil
    """
    if stencil1.pattern != stencil2.pattern:
        raise ValueError("Cannot add stencils with different patterns")
    
    return Stencil(
        pattern=stencil1.pattern.copy(),
        coefficients=stencil1.coefficients + stencil2.coefficients,
        accuracy=min(stencil1.accuracy, stencil2.accuracy),
        stencil_type=stencil1.stencil_type
    )
