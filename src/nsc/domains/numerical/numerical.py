# NSC Numerical Domain - Unified Numerical Dialect
# Finite difference stencils, quadrature, solvers

"""
NSC_numerical - Unified Numerical Domain

This module provides a unified dialect for numerical methods
including finite differences, quadrature, and solvers.

Supported Models:
- CALC: Discrete derivatives
- DISC: Discretization operators
- LEDGER: Convergence invariants
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List


# =============================================================================
# Unified Numerical Type System
# =============================================================================

@dataclass
class FiniteDifferenceStencil:
    """Finite difference stencil coefficients."""
    points: List[int]  # Relative positions
    coefficients: List[float]  # Weights
    order: int  # Derivative order
    accuracy: int  # Order of accuracy


@dataclass
class StencilPoint:
    """Point in a stencil specification."""
    offset: tuple  # (dx, dy, dz) offsets
    weight: float


@dataclass
class QuadratureRule:
    """Numerical integration rule."""
    nodes: List[float]
    weights: List[float]
    domain: str  # 'interval', 'triangle', 'tetrahedron', etc.
    order: int


@dataclass
class LinearSolver:
    """Linear solver specification."""
    method: str  # 'cg', 'gmres', 'direct', etc.
    preconditioner: Optional[str] = None
    max_iterations: int = 1000
    tolerance: float = 1e-10


@dataclass
class NonlinearSolver:
    """Nonlinear solver specification."""
    method: str  # 'newton', 'fixed_point', etc.
    max_iterations: int = 100
    tolerance: float = 1e-8


@dataclass
class TimeStepper:
    """Time integration method."""
    method: str  # 'rk4', 'ssprk3', 'gauss', etc.
    order: int
    stages: int = 1


@dataclass
class ConvergenceInfo:
    """Convergence test information."""
    error: float
    order: float
    grid_sizes: List[int]


# =============================================================================
# Numerical Invariants
# =============================================================================

NUMERICAL_INVARIANTS = {
    'stencil_consistency': {
        'id': 'N:INV.numerical.stencil_consistency',
        'description': 'Stencil coefficients sum correctly',
        'gate_type': 'HARD',
        'receipt_field': 'numerical.stencil_sum'
    },
    'quadrature_exactness': {
        'id': 'N:INV.numerical.quadrature_exactness',
        'description': 'Quadrature exact for polynomials up to order',
        'gate_type': 'SOFT',
        'receipt_field': 'numerical.quadrature_error'
    },
    'solver_convergence': {
        'id': 'N:INV.numerical.solver_convergence',
        'description': 'Iterative solver converged within tolerance',
        'gate_type': 'SOFT',
        'receipt_field': 'numerical.residual_norm'
    },
    'conservation': {
        'id': 'N:INV.numerical.conservation',
        'description': 'Discrete conservation laws satisfied',
        'gate_type': 'HARD',
        'receipt_field': 'numerical.conservation_error'
    }
}


# =============================================================================
# NSC_numerical Dialect Class
# =============================================================================

class NSC_numerical_Dialect:
    """NSC_numerical - Unified Numerical Domain Dialect.
    
    Provides:
    - Finite difference stencils
    - Quadrature rules
    - Linear/nonlinear solvers
    - Time stepping methods
    - Invariant definitions
    """
    
    name = "NSC_numerical"
    version = "1.0"
    
    subdomains = ['stencils', 'quadrature', 'solvers']
    
    mandatory_models = ['CALC', 'DISC', 'LEDGER']
    
    type_hierarchy = {
        'FiniteDifferenceStencil': FiniteDifferenceStencil,
        'StencilPoint': StencilPoint,
        'QuadratureRule': QuadratureRule,
        'LinearSolver': LinearSolver,
        'NonlinearSolver': NonlinearSolver,
        'TimeStepper': TimeStepper,
        'ConvergenceInfo': ConvergenceInfo,
    }
    
    operators = {
        'stencil_apply': 'apply_stencil',
        'stencil_derivative': 'compute_stencil_derivative',
        'quadrature_integrate': 'compute_quadrature',
        'linear_solve': 'solve_linear',
        'nonlinear_solve': 'solve_nonlinear',
        'time_step': 'advance_time',
        'interpolate': 'interpolate_field',
        'restrict': 'restrict_field',
        'prolong': 'prolong_field',
    }
    
    invariants = NUMERICAL_INVARIANTS
    
    def __init__(self):
        """Initialize numerical dialect."""
        pass
    
    def get_type(self, name: str):
        """Get type by name."""
        return self.type_hierarchy.get(name)
    
    def get_operator(self, name: str):
        """Get operator by name."""
        return self.operators.get(name)
    
    def get_invariant(self, name: str):
        """Get invariant by name."""
        return self.invariants.get(name)


# Import subdomains for dialect access
from .stencils import NSC_Stencils_Dialect
from .quadrature import NSC_Quadrature_Dialect
from .solvers import NSC_Solvers_Dialect


# Export singleton
NSC_numerical = NSC_numerical_Dialect()
