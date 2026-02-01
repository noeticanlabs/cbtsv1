# NSC Algebra Domain - Unified Algebra Dialect
# Linear algebra, Lie algebras, tensor operations

"""
NSC_algebra - Unified Algebra Domain

This module provides a unified dialect for algebraic structures
including linear algebra, Lie algebras, and tensor operations.

Supported Models:
- ALG: Algebraic operations
- GEO: Tensor transformations
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List


# =============================================================================
# Unified Algebra Type System
# =============================================================================

@dataclass
class LieAlgebra:
    """Lie algebra structure."""
    structure_constants: Any  # f^ijk
    dimension: int


@dataclass
class LieAlgebraElement:
    """Element of a Lie algebra."""
    components: List[float]
    algebra: str


@dataclass
class ConnectionForm:
    """Connection 1-form on principal bundle."""
    components: List[Any]


@dataclass
class FieldStrength:
    """Field strength 2-form F = dA + A∧A."""
    components: Any


@dataclass
class TensorRank:
    """Tensor rank specification."""
    covariant: int
    contravariant: int


@dataclass
class LinearOperator:
    """Linear operator on vector space."""
    matrix: Any
    domain_dim: int
    codomain_dim: int


@dataclass
class EigenvalueProblem:
    """Eigenvalue problem Av = λv."""
    matrix: Any
    eigenvalues: List[complex]
    eigenvectors: List[Any]


# =============================================================================
# NSC_algebra Dialect Class
# =============================================================================

class NSC_algebra_Dialect:
    """NSC_algebra - Unified Algebra Domain Dialect.
    
    Provides:
    - Linear algebra types and operators
    - Lie algebra types and commutators
    - Tensor operations
    - Eigenvalue/eigenvector computations
    """
    
    name = "NSC_algebra"
    version = "1.0"
    
    subdomains = ['linear', 'lie', 'tensor']
    
    mandatory_models = ['ALG', 'GEO']
    
    type_hierarchy = {
        'LieAlgebra': LieAlgebra,
        'LieAlgebraElement': LieAlgebraElement,
        'ConnectionForm': ConnectionForm,
        'FieldStrength': FieldStrength,
        'TensorRank': TensorRank,
        'LinearOperator': LinearOperator,
        'EigenvalueProblem': EigenvalueProblem,
    }
    
    operators = {
        'lie_bracket': 'compute_lie_bracket',
        'adjoint': 'compute_adjoint',
        'structure_constants': 'compute_structure_constants',
        'tensor_product': 'compute_tensor_product',
        'contraction': 'compute_contraction',
        'symmetrize': 'compute_symmetrization',
        'antisymmetrize': 'compute_antisymmetrization',
        'eigenvalues': 'compute_eigenvalues',
        'eigenvectors': 'compute_eigenvectors',
        'matrix_multiply': 'compute_matrix_multiply',
        'inverse': 'compute_matrix_inverse',
        'determinant': 'compute_determinant',
    }
    
    def __init__(self):
        """Initialize algebra dialect."""
        pass
    
    def get_type(self, name: str):
        """Get type by name."""
        return self.type_hierarchy.get(name)
    
    def get_operator(self, name: str):
        """Get operator by name."""
        return self.operators.get(name)


# Export singleton
NSC_algebra = NSC_algebra_Dialect()
