# NSC Fluids Domain - Unified Fluids Dialect
# Navier-Stokes, Euler, MHD

"""
NSC_fluids - Unified Fluids Domain

This module provides a unified dialect for fluid dynamics
including Navier-Stokes, Euler, and MHD equations.

Supported Models:
- CALC: Time derivatives, advection
- GEO: Divergence, gradient operators
- LEDGER: Conservation invariants
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


# =============================================================================
# Unified Fluids Type System
# =============================================================================

@dataclass
class Velocity:
    """Fluid velocity field v^i."""
    components: List[float] = field(default_factory=list)
    dimension: int = 3


@dataclass
class Pressure:
    """Fluid pressure p."""
    value: float = 0.0


@dataclass
class Density:
    """Fluid density ρ."""
    value: float = 0.0


@dataclass
class Viscosity:
    """Dynamic viscosity μ."""
    value: float = 0.0


@dataclass
class Temperature:
    """Fluid temperature T."""
    value: float = 0.0


@dataclass
class NSVariables:
    """Complete set of Navier-Stokes variables.
    
    Attributes:
        rho: Density
        v: Velocity vector
        p: Pressure
        mu: Dynamic viscosity
        T: Temperature
    """
    rho: Density = field(default_factory=Density)
    v: Velocity = field(default_factory=Velocity)
    p: Pressure = field(default_factory=Pressure)
    mu: Viscosity = field(default_factory=Viscosity)
    T: Temperature = field(default_factory=Temperature)


@dataclass
class StrainRateTensor:
    """Strain rate tensor S_ij = (∂_i v_j + ∂_j v_i)/2."""
    shape: tuple = (3, 3)


@dataclass
class Vorticity:
    """Vorticity ω = ∇×v."""
    components: List[float] = field(default_factory=list)


@dataclass
class Enstrophy:
    """Enstrophy = ½|ω|²."""
    value: float = 0.0


@dataclass
class KineticEnergy:
    """Kinetic energy per unit mass."""
    value: float = 0.0


@dataclass
class DissipationRate:
    """Energy dissipation rate."""
    value: float = 0.0


# =============================================================================
# Fluids Invariants
# =============================================================================

FLUIDS_INVARIANTS = {
    'mass_conservation': {
        'id': 'N:INV.fluids.mass_conservation',
        'description': 'Mass conservation ∂_t ρ + ∇·(ρv) = 0',
        'gate_type': 'HARD',
        'receipt_field': 'fluids.mass_flux'
    },
    'momentum_conservation': {
        'id': 'N:INV.fluids.momentum_conservation',
        'description': 'Momentum conservation ∂_t v + v·∇v = -∇p/ρ + ν∇²v',
        'gate_type': 'HARD',
        'receipt_field': 'fluids.momentum_flux'
    },
    'incompressibility': {
        'id': 'N:INV.fluids.incompressibility',
        'description': 'Divergence-free velocity ∇·v = 0',
        'gate_type': 'SOFT',
        'receipt_field': 'fluids.div_v_max'
    },
    'energy_dissipation': {
        'id': 'N:INV.fluids.energy_dissipation',
        'description': 'Positive energy dissipation rate',
        'gate_type': 'SOFT',
        'receipt_field': 'fluids.dissipation_rate'
    }
}


# =============================================================================
# NSC_fluids Dialect Class
# =============================================================================

class NSC_fluids_Dialect:
    """NSC_fluids - Unified Fluids Domain Dialect.
    
    Provides:
    - Fluids types (Velocity, Pressure, NSVariables)
    - Transport operators (advection, diffusion)
    - Conservation operators
    - Invariant definitions
    """
    
    name = "NSC_fluids"
    version = "1.0"
    
    subdomains = ['navier_stokes', 'euler', 'mhd']
    
    mandatory_models = ['CALC', 'GEO', 'LEDGER']
    
    type_hierarchy = {
        'Velocity': Velocity,
        'Pressure': Pressure,
        'Density': Density,
        'Viscosity': Viscosity,
        'Temperature': Temperature,
        'NSVariables': NSVariables,
        'StrainRateTensor': StrainRateTensor,
        'Vorticity': Vorticity,
        'Enstrophy': Enstrophy,
        'KineticEnergy': KineticEnergy,
        'DissipationRate': DissipationRate,
    }
    
    operators = {
        'advection': 'compute_advection',
        'diffusion': 'compute_diffusion',
        'divergence': 'compute_divergence',
        'gradient_pressure': 'compute_gradient_pressure',
        'strain_rate': 'compute_strain_rate',
        'vorticity': 'compute_vorticity',
        'enstrophy': 'compute_enstrophy',
        'kinetic_energy': 'compute_kinetic_energy',
        'reynolds_number': 'compute_reynolds_number',
        'mach_number': 'compute_mach_number',
    }
    
    invariants = FLUIDS_INVARIANTS
    
    def __init__(self):
        """Initialize fluids dialect."""
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
NSC_fluids = NSC_fluids_Dialect()
