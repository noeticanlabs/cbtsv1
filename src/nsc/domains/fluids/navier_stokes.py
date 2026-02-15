# NSC Fluids - Navier-Stokes Subdomain
# Incompressible fluid dynamics

"""
NSC_NS - Navier-Stokes Domain

This module provides type definitions and operators for incompressible
Navier-Stokes equations with PhaseLoom coherence integration.

Supported Models:
- CALC: Differential operators, PDE residuals
- LEDGER: Divergence-free and energy invariants
- EXEC: VM bytecode generation
- DISC: Finite difference stencils
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

# Type imports from nsc.types
from src.nsc.types import Tensor, Field, Scalar, Vector


# =============================================================================
# NS Type System
# =============================================================================

@dataclass
class Velocity(Vector):
    """Velocity field u(x) ∈ ℝ³.
    
    Represents the fluid velocity at each point in space.
    Used in incompressible Navier-Stokes equations.
    """
    def divergence(self) -> Scalar:
        """Compute div(u) = ∇·u."""
        pass
    
    def curl(self) -> 'Vorticity':
        """Compute ω = ∇×u (vorticity)."""
        return Vorticity()
    
    def kinetic_energy(self) -> Scalar:
        """Compute kinetic energy density ½|u|²."""
        return Scalar(0.0)


@dataclass
class Vorticity(Vector):
    """Vorticity field ω = ∇×u."""
    pass


@dataclass
class Pressure(Scalar):
    """Pressure field p(x).
    
    Enforces incompressibility via ∇p term.
    Defined up to additive constant.
    """
    pass


@dataclass
class Viscosity(Scalar):
    """Dynamic viscosity ν > 0.
    
    Controls viscous dissipation. ν = 0 gives Euler equations.
    """
    pass


@dataclass
class Density(Scalar):
    """Fluid density ρ > 0 (constant for incompressible flow)."""
    pass


@dataclass
class Forcing(Vector):
    """External forcing f(x,t).
    
    Optional source term in momentum equation.
    """
    pass


@dataclass
class NSVariables:
    """Complete set of Navier-Stokes variables.
    
    Attributes:
        u: Velocity field
        p: Pressure field
        nu: Dynamic viscosity
        rho: Density
        f: External forcing (optional)
    """
    u: Velocity
    p: Pressure
    nu: Viscosity
    rho: Density
    f: Optional[Forcing] = None


@dataclass
class DivergenceResidual(Scalar):
    """Divergence constraint residual ε_div = ∇·u.
    
    Must be zero (or below tolerance) for incompressible flow.
    """
    pass


@dataclass
class EnergyResidual(Scalar):
    """Energy balance residual ε_E.
    
    Measures deviation from expected energy change due to
    forcing, dissipation, and boundary fluxes.
    """
    pass


@dataclass
class StrainRateTensor(Tensor):
    """Strain rate tensor S_ij = ½(∇_i u_j + ∇_j u_i)."""
    shape: tuple = (3, 3)


# =============================================================================
# NS Invariants
# =============================================================================

NS_INVARIANTS = {
    'div_free': {
        'id': 'N:INV.ns.div_free',
        'description': 'Velocity divergence-free within tolerance',
        'gate_type': 'HARD',
        'receipt_field': 'residuals.eps_div'
    },
    'energy_nonincreasing': {
        'id': 'N:INV.ns.energy_nonincreasing',
        'description': 'Kinetic energy bounded by forcing/dissipation',
        'gate_type': 'SOFT',
        'receipt_field': 'residuals.eps_energy'
    },
    'cfl_stability': {
        'id': 'N:INV.ns.cfl_stability',
        'description': 'CFL condition satisfied',
        'gate_type': 'SOFT',
        'receipt_field': 'metrics.cfl_number'
    },
    'positivity_pressure': {
        'id': 'N:INV.ns.positivity_pressure',
        'description': 'Pressure positive (physical validity)',
        'gate_type': 'SOFT',
        'receipt_field': 'metrics.p_min'
    }
}


# =============================================================================
# NSC_NS Dialect Class
# =============================================================================

class NSC_NS_Dialect:
    """NSC_NS Dialect for Navier-Stokes.
    
    Provides:
    - NS-specific types (Velocity, Pressure, Vorticity)
    - Fluid operators (advection, diffusion, projection)
    - Invariant definitions (div-free, energy)
    - NIR lowering rules
    """
    
    name = "NSC_fluids.navier_stokes"
    version = "1.0"
    
    mandatory_models = ['CALC', 'LEDGER', 'EXEC']
    optional_models = ['DISC']
    
    type_hierarchy = {
        'Velocity': Velocity,
        'Pressure': Pressure,
        'Viscosity': Viscosity,
        'Density': Density,
        'Forcing': Forcing,
        'Vorticity': Vorticity,
        'DivergenceResidual': DivergenceResidual,
        'EnergyResidual': EnergyResidual,
        'StrainRateTensor': StrainRateTensor,
    }
    
    operators = {
        'advection': 'compute_advection',       # (u·∇)u
        'diffusion': 'compute_diffusion',       # ν∇²u
        'gradient_p': 'compute_pressure_grad',  # ∇p
        'divergence': 'compute_divergence',     # ∇·
        'curl': 'compute_curl',                 # ∇×
        'laplacian': 'compute_laplacian',       # ∇²
        'projection': 'compute_hodge_projection',  # Helmholtz-Hodge
        'strain_rate': 'compute_strain_rate',   # S_ij
        'kinetic_energy': 'compute_ke',         # ½|u|²
        'dissipation': 'compute_dissipation',   # ν|∇u|²
    }
    
    invariants = NS_INVARIANTS
    
    def __init__(self):
        """Initialize NS dialect."""
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


# Export singleton
NSC_NS = NSC_NS_Dialect()


# =============================================================================
# Euler Subdomain (ν = 0 limit)
# =============================================================================

@dataclass
class NSC_Euler_Dialect:
    """NSC_Euler Dialect for ideal fluid (ν = 0).
    
    Special case of Navier-Stokes without viscosity.
    Does NOT claim self-forming barrier property.
    """
    
    name = "NSC_fluids.euler"
    version = "1.0"
    
    mandatory_models = ['CALC', 'LEDGER', 'EXEC']
    
    invariants = {
        'div_free': NS_INVARIANTS['div_free'],
        'circulation_preserved': {
            'id': 'N:INV.euler.circulation',
            'description': 'Kelvin circulation theorem',
            'gate_type': 'SOFT',
            'receipt_field': 'metrics.circulation'
        }
    }


NSC_euler = NSC_Euler_Dialect()
