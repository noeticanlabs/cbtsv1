# NSC Main Package
# Multi-Model Mathematical Linguistics Compiler

"""
NSC - Noetica Symbolic Compiler

A formally specified glyph language for mathematical physics with:
- Multi-model semantics (ALG, CALC, GEO, DISC, LEDGER, EXEC)
- Domain-specific types and operators
- PhaseLoom coherence governance
- Ledger-based audit trail
"""

# Core types
from .types import Scalar, Vector, Tensor, Field

# Domains (unified dialects)
from .domains import (
    NSC_geometry, NSC_fluids, NSC_algebra, 
    NSC_analysis, NSC_numerical, NSC_quantum
)

# Subdomain exports (for backward compatibility)
from .domains.geometry import NSC_GR, NSC_riemann, NSC_YM
from .domains.fluids import NSC_NS
from .domains.algebra import NSC_lie
from .domains.numerical import NSC_stencils

# Models
from .models.ledger import NSC_ledger
from .models.exec.vm import NSC_VM

# Version
__version__ = "2.0.0"

__all__ = [
    # Core types
    'Scalar', 'Vector', 'Tensor', 'Field',
    
    # Unified domains
    'NSC_geometry', 'NSC_fluids', 'NSC_algebra',
    'NSC_analysis', 'NSC_numerical', 'NSC_quantum',
    
    # Subdomains
    'NSC_GR', 'NSC_riemann', 'NSC_YM',
    'NSC_NS', 'NSC_lie', 'NSC_stencils',
    
    # Models
    'NSC_ledger', 'NSC_VM',
    
    # Version
    '__version__',
]
