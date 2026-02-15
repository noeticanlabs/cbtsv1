# NSC Geometry Domain - Unified Geometry Dialect
# Combines GR, Riemannian geometry, and Yang-Mills

"""
NSC_geometry - Unified Geometry Domain

This module provides a unified dialect for differential geometry,
Riemannian geometry, and General Relativity.

Supported Models:
- GEO: Covariant derivative, curvature operators
- CALC: Time derivatives, PDE operators
- LEDGER: Constraint invariants, gate checks
- EXEC: VM bytecode generation
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

# Type imports from nsc.types
from src.nsc.types import Tensor, Field, Scalar, Vector


# =============================================================================
# Unified Geometry Type System
# =============================================================================

@dataclass
class Metric(Tensor):
    """Spatial metric γ_ij (symmetric 2-tensor).
    
    Represents the induced metric on a spatial slice of spacetime.
    Stored in symmetric 6-component format (xx, xy, xz, yy, yz, zz).
    """
    sym6_components: bool = True  # Always stored as sym6
    
    @property
    def signature(self):
        return (3, 0)  # Riemannian (positive definite)
    
    def det(self) -> Scalar:
        """Compute determinant of metric."""
        return Scalar(0.0)
    
    def inverse(self) -> 'Metric':
        """Compute inverse metric γ^ij."""
        return Metric(shape=self.shape)
    
    def christoffel(self) -> 'ChristoffelSymbols':
        """Compute Christoffel symbols from metric."""
        return ChristoffelSymbols()
    
    def ricci(self) -> 'RicciTensor':
        """Compute Ricci tensor from metric."""
        return RicciTensor()


@dataclass
class ExtrinsicK(Tensor):
    """Extrinsic curvature K_ij (symmetric 2-tensor).
    
    Represents the second fundamental form of the spatial slice,
    measuring how the slice is embedded in spacetime.
    """
    sym6_components: bool = True
    trace: Optional[Scalar] = None  # K = γ^ij K_ij


@dataclass  
class ConformalFactor(Scalar):
    """Conformal factor ψ for BSSN variables.
    
    Related to metric by γ_ij = ψ^4 * γ_tilde_ij where γ_tilde_ij
    has unit determinant.
    """
    pass


@dataclass
class ConformalMetric(Tensor):
    """Conformal metric γ_tilde_ij.
    
    Unit determinant metric used in BSSN formulation:
    det(γ_tilde_ij) = 1
    """
    sym6_components: bool = True
    
    def enforce_det_one(self):
        """Enforce unit determinant constraint."""
        pass


@dataclass
class TraceFreeTensor(Tensor):
    """Trace-free symmetric 2-tensor Ã_ij.
    
    Used in BSSN for the evolved extrinsic curvature.
    Satisfies γ̃^ij Ã_ij = 0.
    """
    sym6_components: bool = True


@dataclass
class ConformalConnection(Vector):
    """Conformal connection vector Γ̃^i.
    
    Defined as Γ̃^i = γ̃^jk Γ^i_jk where Γ^i_jk are Christoffel symbols
    of the conformal metric.
    """
    pass


@dataclass
class Lapse(Scalar):
    """Lapse function α.
    
    Measures proper time between spatial slices in 3+1 decomposition.
    Must satisfy α > 0.
    """
    pass


@dataclass
class Shift(Vector):
    """Shift vector β^i.
    
    Describes how spatial coordinates flow between slices.
    """
    pass


@dataclass
class BSSNVariables:
    """Complete set of BSSN variables.
    
    Attributes:
        gamma: Spatial metric (sym6)
        K: Extrinsic curvature (sym6)
        phi: Conformal factor (scalar)
        gamma_tilde: Conformal metric (sym6)
        A_tilde: Trace-free extrinsic curvature (sym6)
        Gamma_tilde: Conformal connection vector
        alpha: Lapse scalar
        beta: Shift vector
    """
    gamma: Metric
    K: ExtrinsicK
    phi: ConformalFactor
    gamma_tilde: ConformalMetric
    A_tilde: TraceFreeTensor
    Gamma_tilde: ConformalConnection
    alpha: Lapse
    beta: Shift
    
    @classmethod
    def from_metric(cls, metric: Metric) -> 'BSSNVariables':
        """Initialize BSSN variables from metric."""
        return cls(
            gamma=metric,
            K=ExtrinsicK(shape=(3, 3)),
            phi=ConformalFactor(value=1.0),
            gamma_tilde=ConformalMetric(shape=(3, 3)),
            A_tilde=TraceFreeTensor(shape=(3, 3)),
            Gamma_tilde=ConformalConnection(shape=(3,)),
            alpha=Lapse(value=1.0),
            beta=Shift(shape=(3,))
        )


@dataclass
class HamiltonianResidual(Scalar):
    """Hamiltonian constraint residual H.
    
    H = R + K^2 - K_ij K^ij - 16πρ
    
    Must be zero for physical solutions.
    """
    pass


@dataclass
class MomentumResidual(Vector):
    """Momentum constraint residual M^i.
    
    M^i = D_j(K^ij - γ^ij K) - 8πj^i
    
    Must be zero for physical solutions.
    """
    pass


@dataclass
class ChristoffelSymbols(Tensor):
    """Christoffel symbols Γ^i_jk (3-index symbols).
    
    Levi-Civita connection coefficients derived from metric.
    """
    shape: tuple = (3, 3, 3)  # (i, j, k)


@dataclass
class RiemannTensor(Tensor):
    """Riemann curvature tensor R^i_jkl.
    
    Measures curvature of connection.
    """
    shape: tuple = (3, 3, 3, 3)  # (i, j, k, l)


@dataclass
class RicciTensor(Tensor):
    """Ricci tensor R_ij.
    
    Contraction of Riemann tensor: R_ij = R^k_ikj.
    """
    shape: tuple = (3, 3)


@dataclass
class RicciScalar(Scalar):
    """Ricci scalar curvature R = γ^ij R_ij."""
    pass


@dataclass
class EinsteinTensor(Tensor):
    """Einstein tensor G_ij = R_ij - ½γ_ij R."""
    shape: tuple = (3, 3)


# =============================================================================
# Unified Geometry Invariants
# =============================================================================

GEOMETRY_INVARIANTS = {
    'hamiltonian_constraint': {
        'id': 'N:INV.geometry.hamiltonian_constraint',
        'description': 'Hamiltonian constraint residual zero',
        'gate_type': 'HARD',
        'receipt_field': 'residuals.eps_H'
    },
    'momentum_constraint': {
        'id': 'N:INV.geometry.momentum_constraint', 
        'description': 'Momentum constraint residual zero',
        'gate_type': 'SOFT',
        'receipt_field': 'residuals.eps_M'
    },
    'det_gamma_positive': {
        'id': 'N:INV.geometry.det_gamma_positive',
        'description': 'Metric determinant positive',
        'gate_type': 'HARD',
        'receipt_field': 'metrics.det_gamma_min'
    },
    'trace_K_real': {
        'id': 'N:INV.geometry.trace_K_real',
        'description': 'Extrinsic curvature trace real',
        'gate_type': 'SOFT',
        'receipt_field': 'metrics.K_trace'
    },
    'bianchi_identity': {
        'id': 'N:INV.geometry.bianchi_identity',
        'description': 'Contracted Bianchi identity ∇_j G^ij = 0',
        'gate_type': 'SOFT',
        'receipt_field': 'residuals.bianchi_residual'
    }
}


# =============================================================================
# NSC_geometry Dialect Class
# =============================================================================

class NSC_geometry_Dialect:
    """NSC_geometry - Unified Geometry Domain Dialect.
    
    Combines GR, Riemannian geometry, and Yang-Mills theories.
    Provides:
    - Geometry-specific types (Metric, ExtrinsicK, BSSNVariables)
    - Geometric operators (Christoffel, Ricci, Riemann)
    - Constraint operators (Hamiltonian, Momentum)
    - Invariant definitions
    - NIR lowering rules
    """
    
    name = "NSC_geometry"
    version = "1.0"
    
    subdomains = ['gr', 'riemann', 'ym']
    
    mandatory_models = ['GEO', 'CALC', 'LEDGER', 'EXEC']
    
    type_hierarchy = {
        'Metric': Metric,
        'ExtrinsicK': ExtrinsicK,
        'ConformalFactor': ConformalFactor,
        'ConformalMetric': ConformalMetric,
        'TraceFreeTensor': TraceFreeTensor,
        'ConformalConnection': ConformalConnection,
        'Lapse': Lapse,
        'Shift': Shift,
        'HamiltonianResidual': HamiltonianResidual,
        'MomentumResidual': MomentumResidual,
        'ChristoffelSymbols': ChristoffelSymbols,
        'RiemannTensor': RiemannTensor,
        'RicciTensor': RicciTensor,
        'RicciScalar': RicciScalar,
        'EinsteinTensor': EinsteinTensor,
    }
    
    operators = {
        'christoffel': 'compute_christoffel',
        'ricci': 'compute_ricci',
        'ricci_scalar': 'compute_ricci_scalar',
        'riemann': 'compute_riemann',
        'hamiltonian_constraint': 'compute_hamiltonian',
        'momentum_constraint': 'compute_momentum',
        'trace_K': 'compute_trace_K',
        'det_metric': 'compute_det',
        'inverse_metric': 'compute_inverse',
        'covariant_derivative': 'compute_covariant_derivative',
        'lie_derivative': 'compute_lie_derivative',
    }
    
    invariants = GEOMETRY_INVARIANTS
    
    def __init__(self):
        """Initialize geometry dialect."""
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
    
    def get_subdomain(self, name: str):
        """Get subdomain by name."""
        subdomains = {
            'gr': NSC_GR_Dialect() if hasattr(NSC_GR_Dialect, '__init__') else None,
            'riemann': NSC_riemann_Dialect() if hasattr(NSC_riemann_Dialect, '__init__') else None,
            'ym': NSC_YM_Dialect() if hasattr(NSC_YM_Dialect, '__init__') else None,
        }
        return subdomains.get(name)


# Import subdomains for dialect access
from .gr import NSC_GR_Dialect
from .riemann import NSC_riemann_Dialect
from .ym import NSC_YM_Dialect


# Export singleton
NSC_geometry = NSC_geometry_Dialect()
