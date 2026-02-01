# NSC Numerical - Finite Difference Stencils
# Discrete derivative operators

"""
NSC_Stencils - Finite Difference Stencils

This module provides stencil definitions for numerical derivatives
used across all NSC domains (GR, fluids, etc.).

Supported Models:
- DISC: Stencil weights, stability constraints
- CALC: Gradient, divergence, curl, laplacian
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


# =============================================================================
# Stencil Types
# =============================================================================

@dataclass
class StencilPoint:
    """Single point in a finite difference stencil.
    
    Attributes:
        offset: Grid offset (i, j, k) from center
        weight: Coefficient for this point
    """
    offset: Tuple[int, int, int]
    weight: float


@dataclass
class FiniteDifferenceStencil:
    """Finite difference stencil for derivatives.
    
    Attributes:
        name: Identifier (e.g., 'central2', 'upwind1', 'compact4')
        derivative_order: Order of derivative (1=first, 2=second)
        accuracy_order: Order of accuracy (2, 4, 6, ...)
        points: List of stencil points
        scheme_type: 'central', 'forward', 'backward', 'compact'
    """
    name: str
    derivative_order: int
    accuracy_order: int
    points: List[StencilPoint]
    scheme_type: str = 'central'
    
    def apply(self, field: np.ndarray, h: float) -> np.ndarray:
        """Apply stencil to field with spacing h."""
        pass
    
    def stability_region(self) -> np.ndarray:
        """Return complex stability region for von Neumann analysis."""
        pass


@dataclass
class StencilFamily:
    """Collection of related stencils for different operators."""
    name: str
    derivative_order: int
    accuracy_order: int
    stencils: Dict[str, FiniteDifferenceStencil] = field(default_factory=dict)


# =============================================================================
# Standard Stencils
# =============================================================================

def central_difference_2() -> FiniteDifferenceStencil:
    """2nd order central difference for first derivative.
    
    f' = (f_{i+1} - f_{i-1}) / (2h)
    """
    return FiniteDifferenceStencil(
        name='central2',
        derivative_order=1,
        accuracy_order=2,
        scheme_type='central',
        points=[
            StencilPoint((-1,),  -0.5),
            StencilPoint((1,),   0.5),
        ]
    )


def central_difference_4() -> FiniteDifferenceStencil:
    """4th order central difference for first derivative.
    
    f' = (-f_{i+2} + 8f_{i+1} - 8f_{i-1} + f_{i-2}) / (12h)
    """
    return FiniteDifferenceStencil(
        name='central4',
        derivative_order=1,
        accuracy_order=4,
        scheme_type='central',
        points=[
            StencilPoint((-2,),  1/12),
            StencilPoint((-1,), -2/3),
            StencilPoint((1,),   2/3),
            StencilPoint((2,),  -1/12),
        ]
    )


def forward_difference_1() -> FiniteDifferenceStencil:
    """1st order forward difference.
    
    f' = (f_{i+1} - f_i) / h
    """
    return FiniteDifferenceStencil(
        name='forward1',
        derivative_order=1,
        accuracy_order=1,
        scheme_type='forward',
        points=[
            StencilPoint((0,), -1.0),
            StencilPoint((1,),  1.0),
        ]
    )


def backward_difference_1() -> FiniteDifferenceStencil:
    """1st order backward difference.
    
    f' = (f_i - f_{i-1}) / h
    """
    return FiniteDifferenceStencil(
        name='backward1',
        derivative_order=1,
        accuracy_order=1,
        scheme_type='backward',
        points=[
            StencilPoint((-1,), -1.0),
            StencilPoint((0,),  1.0),
        ]
    )


def compact_pade_4() -> FiniteDifferenceStencil:
    """4th order compact Pade scheme.
    
    (1/4)f'_{i-1} + f'_i + (1/4)f'_{i+1} = (3/(2h))(f_{i+1} - f_{i-1})
    
    Requires tridiagonal solver (Thomas algorithm).
    """
    return FiniteDifferenceStencil(
        name='compact4',
        derivative_order=1,
        accuracy_order=4,
        scheme_type='compact',
        points=[
            StencilPoint((-1,), 0.25),
            StencilPoint((0,), 1.0),
            StencilPoint((1,), 0.25),
        ]
    )


def second_central_2() -> FiniteDifferenceStencil:
    """2nd order central difference for second derivative.
    
    f'' = (f_{i+1} - 2f_i + f_{i-1}) / h²
    """
    return FiniteDifferenceStencil(
        name='second_central2',
        derivative_order=2,
        accuracy_order=2,
        scheme_type='central',
        points=[
            StencilPoint((-1,),  1.0),
            StencilPoint((0,), -2.0),
            StencilPoint((1,),  1.0),
        ]
    )


def second_central_4() -> FiniteDifferenceStencil:
    """4th order central difference for second derivative.
    
    f'' = (-f_{i+2} + 16f_{i+1} - 30f_i + 16f_{i-1} - f_{i-2}) / (12h²)
    """
    return FiniteDifferenceStencil(
        name='second_central4',
        derivative_order=2,
        accuracy_order=4,
        scheme_type='central',
        points=[
            StencilPoint((-2,), -1/12),
            StencilPoint((-1,),  4/3),
            StencilPoint((0,),  -2.5),
            StencilPoint((1,),   4/3),
            StencilPoint((2,),  -1/12),
        ]
    )


def kreiss_oliger_dissipation() -> FiniteDifferenceStencil:
    """Kreiss-Olger dissipation stencil for stability.
    
    High-frequency damping for hyperbolic systems.
    """
    return FiniteDifferenceStencil(
        name='kreiss_oliger',
        derivative_order=1,
        accuracy_order=4,
        scheme_type='dissipation',
        points=[
            StencilPoint((-2,),  1/16),
            StencilPoint((-1,), -1/2),
            StencilPoint((0,),   0.0),
            StencilPoint((1,),   1/2),
            StencilPoint((2,),  -1/16),
        ]
    )


# =============================================================================
# Tensor Stencils (for derivatives of symmetric 6-tensors)
# =============================================================================

def sym6_derivative_xx() -> FiniteDifferenceStencil:
    """∂_x derivative on sym6 component."""
    return central_difference_2()


def sym6_derivative_xy() -> FiniteDifferenceStencil:
    """∂_x derivative on xy component (mixed)."""
    return central_difference_2()


def sym6_derivative_xz() -> FiniteDifferenceStencil:
    """∂_x derivative on xz component (mixed)."""
    return central_difference_2()


# =============================================================================
# NSC_Stencils Dialect
# =============================================================================

class NSC_Stencils_Dialect:
    """NSC_Stencils Dialect for finite difference methods.
    
    Provides:
    - Stencil definitions for all derivative orders
    - Stability analysis (von Neumann)
    - Compact scheme support
    """
    
    name = "NSC_numerical.stencils"
    version = "1.0"
    
    mandatory_models = ['DISC']
    optional_models = ['CALC']
    
    stencils = {
        # First derivatives
        'central2': central_difference_2,
        'central4': central_difference_4,
        'forward1': forward_difference_1,
        'backward1': backward_difference_1,
        'compact4': compact_pade_4,
        # Second derivatives
        'second_central2': second_central_2,
        'second_central4': second_central_4,
        # Dissipation
        'kreiss_oliger': kreiss_oliger_dissipation,
    }
    
    def get_stencil(self, name: str) -> Optional[FiniteDifferenceStencil]:
        """Get stencil by name."""
        builder = self.stencils.get(name)
        if builder:
            return builder()
        return None
    
    def list_stencils(self) -> List[str]:
        """List available stencil names."""
        return list(self.stencils.keys())


# Export singleton
NSC_stencils = NSC_Stencils_Dialect()


# =============================================================================
# Quadrature Subdomain
# =============================================================================

@dataclass
class QuadratureRule:
    """Numerical integration rule."""
    name: str
    order: int  # Polynomial exactness
    nodes: np.ndarray
    weights: np.ndarray


def gauss_legendre_2() -> QuadratureRule:
    """2-point Gauss-Legendre quadrature."""
    nodes = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    weights = np.array([1.0, 1.0])
    return QuadratureRule('gauss2', 3, nodes, weights)


def gauss_legendre_4() -> QuadratureRule:
    """4-point Gauss-Legendre quadrature."""
    nodes = np.array([
        -np.sqrt(3/7 + 2/7*np.sqrt(6/5)),
        -np.sqrt(3/7 - 2/7*np.sqrt(6/5)),
        np.sqrt(3/7 - 2/7*np.sqrt(6/5)),
        np.sqrt(3/7 + 2/7*np.sqrt(6/5)),
    ])
    weights = np.array([
        (18 - np.sqrt(30)) / 36,
        (18 + np.sqrt(30)) / 36,
        (18 + np.sqrt(30)) / 36,
        (18 - np.sqrt(30)) / 36,
    ])
    return QuadratureRule('gauss4', 7, nodes, weights)


class NSC_Quadrature_Dialect:
    """NSC_Quadrature Dialect for numerical integration."""
    
    name = "NSC_numerical.quadrature"
    version = "1.0"
    
    rules = {
        'gauss2': gauss_legendre_2,
        'gauss4': gauss_legendre_4,
    }


NSC_quadrature = NSC_Quadrature_Dialect()


# =============================================================================
# Solvers Subdomain
# =============================================================================

class NSC_Solvers_Dialect:
    """NSC_Solvers Dialect for numerical solvers.
    
    Provides:
    - Linear solvers (direct, iterative)
    - Eigenvalue solvers
    - Nonlinear solvers
    """
    
    name = "NSC_numerical.solvers"
    version = "1.0"
    
    solvers = {
        'lu': 'Direct LU decomposition',
        'cholesky': 'Cholesky (symmetric positive definite)',
        'cg': 'Conjugate gradient',
        'gmres': 'GMRES (generalized minimal residual)',
        'bicgstab': 'BiCGStab',
        'jacobi': 'Jacobi iteration',
        'gauss_seidel': 'Gauss-Seidel iteration',
    }


NSC_solvers = NSC_Solvers_Dialect()
