# NSC Analysis Domain - Calculus operators

"""
NSC_analysis - Analysis Domain

This module provides operators for calculus operations including
derivatives, integrals, and differential operators.

Supported Models:
- CALC: Time derivatives, partial derivatives
- GEO: Covariant derivative operators
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List


# =============================================================================
# NSC_analysis Dialect Class
# =============================================================================

class NSC_analysis_Dialect:
    """NSC_analysis - Analysis Domain Dialect.
    
    Provides:
    - Derivative operators (partial, covariant, Lie)
    - Integral operators (volume, surface, line)
    - Differential operators (gradient, divergence, curl, Laplacian)
    """
    
    name = "NSC_analysis"
    version = "1.0"
    
    mandatory_models = ['CALC', 'GEO']
    
    operators = {
        'partial_derivative': 'compute_partial_derivative',
        'covariant_derivative': 'compute_covariant_derivative',
        'directional_derivative': 'compute_directional_derivative',
        'lie_derivative': 'compute_lie_derivative',
        'gradient': 'compute_gradient',
        'divergence': 'compute_divergence',
        'curl': 'compute_curl',
        'laplacian': 'compute_laplacian',
        'hessian': 'compute_hessian',
        'volume_integral': 'compute_volume_integral',
        'surface_integral': 'compute_surface_integral',
        'line_integral': 'compute_line_integral',
    }
    
    def __init__(self):
        """Initialize analysis dialect."""
        pass
    
    def get_operator(self, name: str):
        """Get operator by name."""
        return self.operators.get(name)


# Export singleton
NSC_analysis = NSC_analysis_Dialect()
