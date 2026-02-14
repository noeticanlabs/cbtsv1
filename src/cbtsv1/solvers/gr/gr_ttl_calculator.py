"""
Adaptive TTL Calculator for AEONIC Memory System.

Computes TTL values based on simulation parameters to ensure memory retention
scales appropriately with simulation length.

Supports multiple time units:
- Microseconds (us): 10^-6 seconds
- Seconds (s): base unit
- Minutes (m): 60 seconds
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
from enum import Enum
import logging

logger = logging.getLogger('gr_solver.ttl_calculator')


class TimeUnit(Enum):
    """Time unit enumeration for TTL values."""
    MICROSECONDS = "us"  # 10^-6 seconds
    SECONDS = "s"        # Base unit
    MINUTES = "m"        # 60 seconds


def convert_to_seconds(value: float, unit: TimeUnit) -> float:
    """Convert a value from the given time unit to seconds."""
    converters = {
        TimeUnit.MICROSECONDS: 1e-6,
        TimeUnit.SECONDS: 1.0,
        TimeUnit.MINUTES: 60.0,
    }
    return value * converters.get(unit, 1.0)


def convert_from_seconds(seconds: float, unit: TimeUnit) -> float:
    """Convert seconds to the given time unit."""
    converters = {
        TimeUnit.MICROSECONDS: 1e6,
        TimeUnit.SECONDS: 1.0,
        TimeUnit.MINUTES: 1.0 / 60.0,
    }
    return seconds * converters.get(unit, 1.0)


@dataclass
class TTLValue:
    """
    A TTL value with associated time unit.
    
    Provides convenient conversion between time units while maintaining
    the original unit for display purposes.
    """
    value: float
    unit: TimeUnit = TimeUnit.SECONDS
    
    def __post_init__(self):
        """Validate and normalize TTL value."""
        if self.value < 0:
            raise ValueError(f"TTL value must be non-negative, got {self.value}")
    
    @property
    def seconds(self) -> float:
        """Get value in seconds."""
        return convert_to_seconds(self.value, self.unit)
    
    @property
    def microseconds(self) -> float:
        """Get value in microseconds."""
        return convert_from_seconds(self.seconds, TimeUnit.MICROSECONDS)
    
    @property
    def minutes(self) -> float:
        """Get value in minutes."""
        return convert_from_seconds(self.seconds, TimeUnit.MINUTES)
    
    def to_unit(self, unit: TimeUnit) -> float:
        """Get value in the specified unit."""
        if unit == TimeUnit.MICROSECONDS:
            return self.microseconds
        elif unit == TimeUnit.MINUTES:
            return self.minutes
        return self.value
    
    def __repr__(self):
        return f"TTLValue({self.value}{self.unit.value})"
    
    def __str__(self):
        return f"{self.value} {self.unit.value}"


@dataclass
class AdaptiveTTLs:
    """
    Container for computed adaptive TTL values.
    
    Supports multiple time units: microseconds, seconds, and minutes.
    All values are stored in seconds internally for consistency.
    """
    # M_solve TTLs (attempt receipts) - stored in seconds
    msolve_ttl_s: float  # Short TTL in seconds
    msolve_ttl_l: float  # Long TTL in seconds
    
    # M_step TTLs (accepted steps)
    mstep_ttl_s: float
    mstep_ttl_l: float
    
    # M_orch TTLs (canon promotions)
    morch_ttl_s: float
    morch_ttl_l: float
    
    # Time unit for display (default: seconds)
    display_unit: TimeUnit = TimeUnit.SECONDS
    
    def __post_init__(self):
        """Validate and ensure non-negative values."""
        for field_name in ['msolve_ttl_s', 'msolve_ttl_l', 'mstep_ttl_s', 
                          'mstep_ttl_l', 'morch_ttl_s', 'morch_ttl_l']:
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative, got {value}")
    
    # Short TTL properties in different units
    @property
    def msolve_ttl_s_us(self) -> float:
        """Get M_solve short TTL in microseconds."""
        return self.msolve_ttl_s * 1e6
    
    @property
    def msolve_ttl_s_s(self) -> float:
        """Get M_solve short TTL in seconds."""
        return self.msolve_ttl_s
    
    @property
    def msolve_ttl_s_m(self) -> float:
        """Get M_solve short TTL in minutes."""
        return self.msolve_ttl_s / 60.0
    
    # Long TTL properties in different units
    @property
    def msolve_ttl_l_us(self) -> float:
        """Get M_solve long TTL in microseconds."""
        return self.msolve_ttl_l * 1e6
    
    @property
    def msolve_ttl_l_s(self) -> float:
        """Get M_solve long TTL in seconds."""
        return self.msolve_ttl_l
    
    @property
    def msolve_ttl_l_m(self) -> float:
        """Get M_solve long TTL in minutes."""
        return self.msolve_ttl_l / 60.0
    
    # M_step short TTL properties
    @property
    def mstep_ttl_s_us(self) -> float:
        return self.mstep_ttl_s * 1e6
    
    @property
    def mstep_ttl_s_s(self) -> float:
        return self.mstep_ttl_s
    
    @property
    def mstep_ttl_s_m(self) -> float:
        return self.mstep_ttl_s / 60.0
    
    # M_step long TTL properties
    @property
    def mstep_ttl_l_us(self) -> float:
        return self.mstep_ttl_l * 1e6
    
    @property
    def mstep_ttl_l_s(self) -> float:
        return self.mstep_ttl_l
    
    @property
    def mstep_ttl_l_m(self) -> float:
        return self.mstep_ttl_l / 60.0
    
    # M_orch short TTL properties
    @property
    def morch_ttl_s_us(self) -> float:
        return self.morch_ttl_s * 1e6
    
    @property
    def morch_ttl_s_s(self) -> float:
        return self.morch_ttl_s
    
    @property
    def morch_ttl_s_m(self) -> float:
        return self.morch_ttl_s / 60.0
    
    # M_orch long TTL properties
    @property
    def morch_ttl_l_us(self) -> float:
        return self.morch_ttl_l * 1e6
    
    @property
    def morch_ttl_l_s(self) -> float:
        return self.morch_ttl_l
    
    @property
    def morch_ttl_l_m(self) -> float:
        return self.morch_ttl_l / 60.0
    
    def get_msolve_short(self, unit: TimeUnit) -> float:
        """Get M_solve short TTL in specified unit."""
        if unit == TimeUnit.MICROSECONDS:
            return self.msolve_ttl_s_us
        elif unit == TimeUnit.MINUTES:
            return self.msolve_ttl_s_m
        return self.msolve_ttl_s_s
    
    def get_msolve_long(self, unit: TimeUnit) -> float:
        """Get M_solve long TTL in specified unit."""
        if unit == TimeUnit.MICROSECONDS:
            return self.msolve_ttl_l_us
        elif unit == TimeUnit.MINUTES:
            return self.msolve_ttl_l_m
        return self.msolve_ttl_l_s
    
    def get_mstep_short(self, unit: TimeUnit) -> float:
        """Get M_step short TTL in specified unit."""
        if unit == TimeUnit.MICROSECONDS:
            return self.mstep_ttl_s_us
        elif unit == TimeUnit.MINUTES:
            return self.mstep_ttl_s_m
        return self.mstep_ttl_s_s
    
    def get_mstep_long(self, unit: TimeUnit) -> float:
        """Get M_step long TTL in specified unit."""
        if unit == TimeUnit.MICROSECONDS:
            return self.mstep_ttl_l_us
        elif unit == TimeUnit.MINUTES:
            return self.mstep_ttl_l_m
        return self.mstep_ttl_l_s
    
    def get_morch_short(self, unit: TimeUnit) -> float:
        """Get M_orch short TTL in specified unit."""
        if unit == TimeUnit.MICROSECONDS:
            return self.morch_ttl_s_us
        elif unit == TimeUnit.MINUTES:
            return self.morch_ttl_s_m
        return self.morch_ttl_s_s
    
    def get_morch_long(self, unit: TimeUnit) -> float:
        """Get M_orch long TTL in specified unit."""
        if unit == TimeUnit.MICROSECONDS:
            return self.morch_ttl_l_us
        elif unit == TimeUnit.MINUTES:
            return self.morch_ttl_l_m
        return self.morch_ttl_l_s
    
    def __repr__(self):
        return (f"AdaptiveTTLs(msolve=({self.msolve_ttl_s_us:.1f}us/{self.msolve_ttl_l_s:.2f}s, "
                f"mstep=({self.mstep_ttl_s_s:.2f}s/{self.mstep_ttl_l_m:.2f}m, "
                f"morch=({self.morch_ttl_s_m:.2f}m/{self.morch_ttl_l_s:.2f}s))")
    
    def to_dict(self, unit: TimeUnit = None) -> Dict[str, float]:
        """
        Convert to dictionary with values in specified unit.
        
        Args:
            unit: Time unit for values (defaults to display_unit)
        
        Returns:
            Dictionary with TTL values
        """
        unit = unit or self.display_unit
        return {
            'msolve_ttl_s': self.get_msolve_short(unit),
            'msolve_ttl_l': self.get_msolve_long(unit),
            'mstep_ttl_s': self.get_mstep_short(unit),
            'mstep_ttl_l': self.get_mstep_long(unit),
            'morch_ttl_s': self.get_morch_short(unit),
            'morch_ttl_l': self.get_morch_long(unit),
        }


class TTLCalculator:
    """
    Computes adaptive TTL values based on simulation parameters.
    
    The TTL scales with:
    - Total simulation time (t_end)
    - Average timestep (dt_avg)
    - Grid size (N) - affects computational cost
    - Problem type - affects retention requirements
    
    All TTLs are computed in seconds, with automatic conversion to
    microseconds and minutes available via property accessors.
    """
    
    # TTL scaling factors (fraction of total simulation time)
    MSOLVE_SCALE = 0.05    # 5% of total simulation for attempts
    MSTEP_SCALE = 0.10     # 10% of total simulation for steps
    MORCH_SCALE = 0.50     # 50% of total simulation for canon
    
    # Minimum TTLs (in seconds)
    MIN_MSOLVE_S = 1e-6        # 1 microsecond minimum
    MIN_MSOLVE_L = 1.0         # 1 second minimum
    MIN_MSTEP_S = 10.0         # 10 seconds minimum
    MIN_MSTEP_L = 600.0        # 10 minutes minimum
    MIN_MORCH_S = 3600.0       # 1 hour minimum
    MIN_MORCH_L = 86400.0      # 1 day minimum
    
    # Maximum TTLs (in seconds) - prevent unbounded growth
    MAX_MSOLVE_S = 60.0        # 1 minute maximum
    MAX_MSOLVE_L = 3600.0      # 1 hour maximum
    MAX_MSTEP_S = 3600.0       # 1 hour maximum
    MAX_MSTEP_L = 86400.0      # 1 day maximum
    MAX_MORCH_S = 604800.0     # 1 week maximum
    MAX_MORCH_L = 2592000.0    # 30 days maximum
    
    # Grid size scaling factor (larger grids = higher recompute cost = longer retention)
    GRID_SIZE_REF = 64  # Reference grid size
    
    # Problem type multipliers
    PROBLEM_MULTIPLIERS = {
        'standard': 1.0,
        'long_run': 2.0,       # Longer retention for extended simulations
        'high_frequency': 0.5,  # Shorter retention for high-frequency data
        'critical': 3.0,        # Maximum retention for critical problems
        'transient': 0.3,       # Shorter retention for transient analysis
    }
    
    def __init__(
        self,
        t_end: float,
        dt_avg: float,
        N: int,
        problem_type: str = 'standard',
        tick_rate: int = 1,
        output_unit: TimeUnit = TimeUnit.SECONDS
    ):
        """
        Initialize the TTL calculator.
        
        Args:
            t_end: Final simulation time (in seconds)
            dt_avg: Average timestep (in seconds)
            N: Grid size (effective, e.g., Nx*Ny*Nz or max dimension)
            problem_type: Type of simulation ('standard', 'long_run', 'high_frequency', 'critical', 'transient')
            tick_rate: Ticks per second (default 1 tick per second)
            output_unit: Default time unit for output TTLs
        """
        self.t_end = t_end
        self.dt_avg = dt_avg
        self.N = N
        self.problem_type = problem_type
        self.tick_rate = tick_rate
        self.output_unit = output_unit
        
        # Compute estimated number of steps
        self.estimated_steps = int(t_end / dt_avg) if dt_avg > 0 else 0
        
        # Compute grid size multiplier
        self.grid_multiplier = max(1.0, (N / self.GRID_SIZE_REF) ** 0.5)
        
        # Get problem type multiplier
        self.problem_multiplier = self.PROBLEM_MULTIPLIERS.get(
            problem_type, self.PROBLEM_MULTIPLIERS['standard']
        )
        
        logger.debug(f"TTLCalculator initialized: t_end={t_end}s, dt_avg={dt_avg}s, "
                     f"N={N}, problem_type={problem_type}, estimated_steps={self.estimated_steps}")
    
    def compute_ttls(self) -> AdaptiveTTLs:
        """
        Compute adaptive TTL values for all memory tiers.
        
        Returns:
            AdaptiveTTLs object with TTL values in seconds
        """
        # Compute base TTLs in seconds
        base_msolve = self.t_end * self.MSOLVE_SCALE * self.problem_multiplier
        base_mstep = self.t_end * self.MSTEP_SCALE * self.problem_multiplier
        base_morch = self.t_end * self.MORCH_SCALE * self.problem_multiplier
        
        # Apply grid size scaling (larger grids = longer retention due to higher recompute cost)
        grid_factor = self.grid_multiplier
        base_msolve *= grid_factor
        base_mstep *= grid_factor
        base_morch *= grid_factor
        
        # Compute long TTL as 4x short TTL
        msolve_ttl_s = base_msolve
        msolve_ttl_l = base_msolve * 4
        mstep_ttl_s = base_mstep
        mstep_ttl_l = base_mstep * 4
        morch_ttl_s = base_morch
        morch_ttl_l = base_morch * 4
        
        # Apply min/max bounds for M_solve
        msolve_ttl_s = max(self.MIN_MSOLVE_S, min(msolve_ttl_s, self.MAX_MSOLVE_S))
        msolve_ttl_l = max(self.MIN_MSOLVE_L, min(msolve_ttl_l, self.MAX_MSOLVE_L))
        
        # Apply min/max bounds for M_step
        mstep_ttl_s = max(self.MIN_MSTEP_S, min(mstep_ttl_s, self.MAX_MSTEP_S))
        mstep_ttl_l = max(self.MIN_MSTEP_L, min(mstep_ttl_l, self.MAX_MSTEP_L))
        
        # Apply min/max bounds for M_orch
        morch_ttl_s = max(self.MIN_MORCH_S, min(morch_ttl_s, self.MAX_MORCH_S))
        morch_ttl_l = max(self.MIN_MORCH_L, min(morch_ttl_l, self.MAX_MORCH_L))
        
        logger.debug(f"Computed TTLs: M_solve=({msolve_ttl_s:.6f}s/{msolve_ttl_l:.2f}s), "
                     f"M_step=({mstep_ttl_s:.2f}s/{mstep_ttl_l:.2f}s), "
                     f"M_orch=({morch_ttl_s:.2f}s/{morch_ttl_l:.2f}s)")
        
        return AdaptiveTTLs(
            msolve_ttl_s=msolve_ttl_s,
            msolve_ttl_l=msolve_ttl_l,
            mstep_ttl_s=mstep_ttl_s,
            mstep_ttl_l=mstep_ttl_l,
            morch_ttl_s=morch_ttl_s,
            morch_ttl_l=morch_ttl_l,
            display_unit=self.output_unit
        )
    
    def get_ttl_value(self, ttl_type: str, duration: float, unit: TimeUnit = None) -> TTLValue:
        """
        Get a TTLValue for a specific TTL type.
        
        Args:
            ttl_type: Type of TTL ('msolve_s', 'msolve_l', 'mstep_s', 'mstep_l', 'morch_s', 'morch_l')
            duration: Duration value
            unit: Time unit for the duration
        
        Returns:
            TTLValue object
        """
        unit = unit or TimeUnit.SECONDS
        return TTLValue(value=duration, unit=unit)
    
    def get_static_fallback(self) -> AdaptiveTTLs:
        """
        Return static TTL fallback values for backward compatibility.
        
        Returns:
            AdaptiveTTLs with static TTL values in seconds
        """
        return AdaptiveTTLs(
            msolve_ttl_s=self.MIN_MSOLVE_S,      # 1e-6 seconds
            msolve_ttl_l=self.MIN_MSOLVE_L,      # 1 second
            mstep_ttl_s=self.MIN_MSTEP_S,        # 10 seconds
            mstep_ttl_l=self.MIN_MSTEP_L,        # 10 minutes
            morch_ttl_s=self.MIN_MORCH_S,        # 1 hour
            morch_ttl_l=self.MIN_MORCH_L,        # 1 day
            display_unit=self.output_unit
        )
    
    @classmethod
    def create_from_simulation(
        cls,
        t_end: float,
        dt_avg: float,
        N: int,
        problem_type: str = 'standard'
    ) -> 'TTLCalculator':
        """
        Factory method to create TTLCalculator from simulation parameters.
        
        Args:
            t_end: Final simulation time
            dt_avg: Average timestep
            N: Grid size
            problem_type: Type of simulation
        
        Returns:
            TTLCalculator instance
        """
        return cls(t_end=t_end, dt_avg=dt_avg, N=N, problem_type=problem_type)


def compute_adaptive_ttls(
    t_end: float,
    dt_avg: float,
    N: int,
    problem_type: str = 'standard'
) -> AdaptiveTTLs:
    """
    Convenience function to compute adaptive TTLs.
    
    Args:
        t_end: Final simulation time
        dt_avg: Average timestep
        N: Grid size
        problem_type: Type of simulation
    
    Returns:
        AdaptiveTTLs object with computed TTL values
    """
    calculator = TTLCalculator.create_from_simulation(
        t_end=t_end, dt_avg=dt_avg, N=N, problem_type=problem_type
    )
    return calculator.compute_ttls()
