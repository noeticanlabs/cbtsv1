# NSC Algebra - Lie Algebra Subdomain
# Lie groups, algebras, representations

"""
NSC_LIE - Lie Algebra Domain

This module provides type definitions and operators for Lie algebras,
used in Yang-Mills theory, particle physics, and differential geometry.

Supported Models:
- ALG: Commutator, adjoint, representation operations
- GEO: Covariant derivative on principal bundles
- CALC: Lie derivative operators
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np


# =============================================================================
# Lie Algebra Type System
# =============================================================================

@dataclass
class LieAlgebra:
    """Lie algebra representation.
    
    Attributes:
        name: Name (e.g., 'su(2)', 'su(3)', 'so(3)')
        dimension: Dimension of the algebra
        structure_constants: f^{ijk} where [T^i, T^j] = f^{ijk} T^k
        generators: Matrix generators (if available)
    """
    name: str = ''
    dimension: int = 0
    structure_constants: Optional[np.ndarray] = None
    generators: Optional[List[np.ndarray]] = None


@dataclass
class LieAlgebraElement:
    """Element of a Lie algebra.
    
    Represented as coefficients in basis:
    X = X^a T_a where T_a are generators.
    """
    components: List[float] = field(default_factory=list)
    algebra: Optional[LieAlgebra] = None
    
    def commutator(self, other: 'LieAlgebraElement') -> 'LieAlgebraElement':
        """Compute [X, Y] = XY - YX."""
        return LieAlgebraElement()
    
    def adjoint(self) -> np.ndarray:
        """Compute adjoint representation matrix (ad_X)_{ab} = f^{acb} X_c."""
        return np.array([])


@dataclass
class LieGroupElement:
    """Element of a Lie group.
    
    Exponentiation of Lie algebra element:
    g = exp(X) where X ∈ 𝔤
    """
    algebra_element: Optional[LieAlgebraElement] = None
    
    def multiply(self, other: 'LieGroupElement') -> 'LieGroupElement':
        """Group multiplication."""
        return LieGroupElement()
    
    def inverse(self) -> 'LieGroupElement':
        """Group inverse."""
        return LieGroupElement()
    
    def log(self) -> LieAlgebraElement:
        """Logarithm: return X such that exp(X) = g."""
        return LieAlgebraElement()


@dataclass
class ConnectionForm:
    """Connection 1-form A on principal bundle.
    
    A = A^a ⊗ T_a where T_a are gauge group generators.
    """
    components: np.ndarray = field(default_factory=lambda: np.array([]))
    algebra: Optional[LieAlgebra] = None


@dataclass
class FieldStrength:
    """Field strength 2-form F = dA + ½[A, A].
    
    Curvature of the connection:
    F = dA + A ∧ A
    """
    components: np.ndarray = field(default_factory=lambda: np.array([]))
    algebra: Optional[LieAlgebra] = None


@dataclass
class GaugeCovariantDerivative:
    """Gauge covariant derivative D = d + A.
    
    Acting on matter fields in representation ρ:
    D_μ ψ = ∂_μ ψ + ρ(A_μ) ψ
    """
    connection: Optional[ConnectionForm] = None
    representation: str = ''


@dataclass
class GaussLawResidual:
    """Gauss law constraint residual.
    
    D_i E^i = 0 for Yang-Mills theory.
    """
    value: float = 0.0


@dataclass
class BianchiResidual:
    """Bianchi identity residual D_{[i}F_{jk]} = 0."""
    value: float = 0.0


# =============================================================================
# YM Invariants
# =============================================================================

YM_INVARIANTS = {
    'gauss_law': {
        'id': 'N:INV.ym.gauss_law',
        'description': 'D_i E^i = 0 (charge conservation)',
        'gate_type': 'HARD',
        'receipt_field': 'residuals.eps_G'
    },
    'bianchi_identity': {
        'id': 'N:INV.ym.bianchi',
        'description': 'D_{[i}F_{jk]} = 0',
        'gate_type': 'HARD',
        'receipt_field': 'residuals.eps_BI'
    },
    'gauge_condition': {
        'id': 'N:INV.ym.gauge_condition',
        'description': 'Gauge condition residual bounded',
        'gate_type': 'SOFT',
        'receipt_field': 'metrics.gauge_residual'
    },
    'energy_consistency': {
        'id': 'N:INV.ym.energy_consistency',
        'description': 'Energy bounded/monotone',
        'gate_type': 'SOFT',
        'receipt_field': 'residuals.eps_energy'
    }
}


# =============================================================================
# Standard Lie Algebras
# =============================================================================

def su2() -> LieAlgebra:
    """su(2) - 3-dimensional Lie algebra (isomorphic to so(3))."""
    generators = [
        np.array([[0, 1], [1, 0]], dtype=np.complex128) / 2,
        np.array([[0, -1j], [1j, 0]], dtype=np.complex128) / 2,
        np.array([[1, 0], [0, -1]], dtype=np.complex128) / 2,
    ]
    return LieAlgebra(name='su(2)', dimension=3, generators=generators)


def su3() -> LieAlgebra:
    """su(3) - 8-dimensional Lie algebra (QCD gauge group)."""
    return LieAlgebra(name='su(3)', dimension=8)


def so3() -> LieAlgebra:
    """so(3) - 3-dimensional rotation algebra."""
    generators = [
        np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]]),
        np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]]),
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]),
    ]
    return LieAlgebra(name='so(3)', dimension=3, generators=generators)


# =============================================================================
# NSC_LIE Dialect Class
# =============================================================================

class NSC_LIE_Dialect:
    """NSC_LIE Dialect for Lie algebras.
    
    Provides:
    - Lie algebra types (algebra, element, group)
    - Algebra operations (commutator, adjoint)
    - Gauge theory types (connection, field strength)
    - YM invariants
    """
    
    name = "NSC_algebra.lie"
    version = "1.0"
    
    mandatory_models = ['ALG', 'GEO']
    optional_models = ['CALC', 'LEDGER']
    
    type_hierarchy = {
        'LieAlgebra': LieAlgebra,
        'LieAlgebraElement': LieAlgebraElement,
        'LieGroupElement': LieGroupElement,
        'ConnectionForm': ConnectionForm,
        'FieldStrength': FieldStrength,
        'GaugeCovariantDerivative': GaugeCovariantDerivative,
        'GaussLawResidual': GaussLawResidual,
        'BianchiResidual': BianchiResidual,
    }
    
    operators = {
        'commutator': 'compute_commutator',        # [X, Y]
        'adjoint': 'compute_adjoint',              # ad_X
        'exponential': 'compute_exp',              # exp(X)
        'logarithm': 'compute_log',                # log(g)
        'structure_constants': 'get_structure',    # f^{ijk}
        'gauge_covariant': 'compute_D',            # D = d + A
        'field_strength': 'compute_F',             # F = dA + A∧A
    }
    
    invariants = YM_INVARIANTS
    
    def __init__(self, algebra: Optional[LieAlgebra] = None):
        """Initialize LIE dialect with optional algebra."""
        self.algebra = algebra
    
    def get_type(self, name: str):
        """Get type by name."""
        return self.type_hierarchy.get(name)
    
    def get_operator(self, name: str):
        """Get operator by name."""
        return self.operators.get(name)
    
    def get_invariant(self, name: str):
        """Get invariant by name."""
        return self.invariants.get(name)


# Export singleton
NSC_lie = NSC_LIE_Dialect()
