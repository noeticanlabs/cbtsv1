# NSC Compatibility Layer
# Backward compatibility aliases for old module paths

"""
This module provides backward compatibility aliases for the NSC
dialect reorganization. Old import paths will continue to work
during the deprecation period.

Usage:
    from nsc.compat import NSC_GR  # Old style
    from nsc.domains.geometry.gr import NSC_GR  # New style
"""

# =============================================================================
# Old Dialect Aliases (Backward Compatibility)
# =============================================================================

# Old: from nsc.alg_types
# New: from nsc.domains.algebra.tensor.types
from nsc.domains.algebra.tensor import NSC_tensor as alg_types

# Old: from nsc.ledger_gates
# New: from nsc.models.ledger import NSC_ledger
from nsc.models.ledger import NSC_ledger as ledger_gates

# Old: from nsc.stencils
# New: from nsc.domains.numerical.stencils import NSC_stencils
from nsc.domains.numerical.stencils import NSC_stencils as stencils

# Old: from nsc.exec_vm
# New: from nsc.models.exec.vm import NSC_VM
from nsc.models.exec.vm import NSC_VM

# Old: from nsc.quadrature
# New: from nsc.domains.numerical.quadrature
from nsc.domains.numerical.quadrature import NSC_quadrature as quadrature

# =============================================================================
# Dialect Compatibility
# =============================================================================

# Old dialect names -> New modules
NSC_GR = "Use nsc.domains.geometry.gr.NSC_GR"
NSC_NS = "Use nsc.domains.fluids.navier_stokes.NSC_NS"
NSC_YM = "Use nsc.domains.geometry.ym"
NSC_Time = "Use nsc.models.ledger and nsc.models.exec"


def __getattr__(name):
    """Provide dynamic access to deprecated names."""
    import warnings
    
    deprecation_warnings = {
        'NSC_GR': 'Use "from nsc.domains.geometry.gr import NSC_GR" instead',
        'NSC_NS': 'Use "from nsc.domains.fluids.navier_stokes import NSC_NS" instead',
        'NSC_YM': 'Use "from nsc.domains.geometry.ym import NSC_YM" instead',
        'NSC_Time': 'Use "from nsc.models.ledger import NSC_ledger" instead',
    }
    
    if name in deprecation_warnings:
        warnings.warn(
            f"'{name}' is deprecated. {deprecation_warnings[name]}",
            DeprecationWarning,
            stacklevel=2
        )
        
        if name == 'NSC_GR':
            from nsc.domains.geometry.gr import NSC_GR
            return NSC_GR
        elif name == 'NSC_NS':
            from nsc.domains.fluids.navier_stokes import NSC_NS
            return NSC_NS
        elif name == 'NSC_YM':
            from nsc.domains.geometry.ym import NSC_YM
            return NSC_YM
        elif name == 'NSC_Time':
            from nsc.models.ledger import NSC_ledger
            from nsc.models.exec.vm import NSC_VM as NSC_exec # Changed import for NSC_exec
            return (NSC_ledger, NSC_exec)
    
    raise AttributeError(f"Module 'nsc.compat' has no attribute '{name}'")


# =============================================================================
# Import Path Compatibility
# =============================================================================

# These allow old-style imports to work:
# import nsc.alg_types -> import nsc.domains.algebra.tensor.types

class _CompatModule:
    """Helper class for compatibility imports."""
    
    @property
    def alg_types(self):
        from nsc.domains.algebra.tensor import NSC_tensor
        return NSC_tensor
    
    @property
    def ledger_gates(self):
        from nsc.models.ledger import NSC_ledger
        return NSC_ledger
    
    @property
    def stencils(self):
        from nsc.domains.numerical.stencils import NSC_stencils
        return NSC_stencils
    
    @property
    def exec_vm(self):
        from nsc.models.exec.vm import NSC_VM
        return NSC_VM
    
    @property
    def quadrature(self):
        from nsc.domains.numerical.quadrature import NSC_quadrature
        return NSC_quadrature


# Create module instance for attribute-based access
_compat = _CompatModule()