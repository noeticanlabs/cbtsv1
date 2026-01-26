"""
Adaptive TTL Calculator for AEONIC Memory System.

Computes TTL values based on simulation parameters to ensure memory retention
scales appropriately with simulation length.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger('gr_solver.ttl_calculator')


@dataclass
class AdaptiveTTLs:
    """Container for computed adaptive TTL values."""
    # M_solve TTLs (attempt receipts)
    msolve_ttl_s: int  # Short TTL in ticks
    msolve_ttl_l: int  # Long TTL in ticks
    
    # M_step TTLs (accepted steps)
    mstep_ttl_s: int
    mstep_ttl_l: int
    
    # M_orch TTLs (canon promotions)
    morch_ttl_s: int
    morch_ttl_l: int


class TTLCalculator:
    """
    Computes adaptive TTL values based on simulation parameters.
    
    The TTL scales with:
    - Total simulation time (t_end)
    - Average timestep (dt_avg)
    - Grid size (N) - affects computational cost
    - Problem type - affects retention requirements
    """
    
    # TTL scaling factors (fraction of total simulation time)
    MSOLVE_SCALE = 0.05    # 5% of total simulation for attempts
    MSTEP_SCALE = 0.10     # 10% of total simulation for steps
    MORCH_SCALE = 0.50     # 50% of total simulation for canon
    
    # Minimum TTLs (in ticks) - fallback to prevent overly aggressive expiration
    MIN_MSOLVE_S = 3600    # 1 hour minimum
    MIN_MSOLVE_L = 86400   # 1 day minimum
    MIN_MSTEP_S = 36000    # 10 hours minimum
    MIN_MSTEP_L = 604800   # 1 week minimum
    MIN_MORCH_S = 2592000  # 30 days minimum
    MIN_MORCH_L = 2592000  # 30 days minimum
    
    # Maximum TTLs (in ticks) - prevent unbounded growth
    MAX_MSOLVE_S = 86400   # 1 day maximum
    MAX_MSOLVE_L = 604800  # 1 week maximum
    MAX_MSTEP_S = 604800   # 1 week maximum
    MAX_MSTEP_L = 2592000  # 30 days maximum
    MAX_MORCH_S = 2592000  # 30 days maximum
    MAX_MORCH_L = 31536000 # 1 year maximum
    
    # Grid size scaling factor (larger grids = higher recompute cost = longer retention)
    GRID_SIZE_REF = 64  # Reference grid size
    
    # Problem type multipliers
    PROBLEM_MULTIPLIERS = {
        'standard': 1.0,
        'long_run': 2.0,      # Longer retention for extended simulations
        'high_frequency': 0.5, # Shorter retention for high-frequency data
        'critical': 3.0,       # Maximum retention for critical problems
        'transient': 0.3,      # Shorter retention for transient analysis
    }
    
    def __init__(
        self,
        t_end: float,
        dt_avg: float,
        N: int,
        problem_type: str = 'standard',
        tick_rate: int = 1
    ):
        """
        Initialize the TTL calculator.
        
        Args:
            t_end: Final simulation time
            dt_avg: Average timestep
            N: Grid size (effective, e.g., Nx*Ny*Nz or max dimension)
            problem_type: Type of simulation ('standard', 'long_run', 'high_frequency', 'critical', 'transient')
            tick_rate: Ticks per unit time (default 1 tick per unit)
        """
        self.t_end = t_end
        self.dt_avg = dt_avg
        self.N = N
        self.problem_type = problem_type
        self.tick_rate = tick_rate
        
        # Compute estimated number of steps
        self.estimated_steps = int(t_end / dt_avg) if dt_avg > 0 else 0
        
        # Compute grid size multiplier
        self.grid_multiplier = max(1.0, (N / self.GRID_SIZE_REF) ** 0.5)
        
        # Get problem type multiplier
        self.problem_multiplier = self.PROBLEM_MULTIPLIERS.get(
            problem_type, self.PROBLEM_MULTIPLIERS['standard']
        )
        
        logger.debug(f"TTLCalculator initialized: t_end={t_end}, dt_avg={dt_avg}, "
                     f"N={N}, problem_type={problem_type}, estimated_steps={self.estimated_steps}")
    
    def compute_ttls(self) -> AdaptiveTTLs:
        """
        Compute adaptive TTL values for all memory tiers.
        
        Returns:
            AdaptiveTTLs object with TTL values in ticks
        """
        # Convert simulation time to ticks
        total_ticks = int(self.t_end * self.tick_rate)
        
        # Compute base TTLs (in time units, will convert to ticks)
        base_msolve = self.t_end * self.MSOLVE_SCALE * self.problem_multiplier
        base_mstep = self.t_end * self.MSTEP_SCALE * self.problem_multiplier
        base_morch = self.t_end * self.MORCH_SCALE * self.problem_multiplier
        
        # Apply grid size scaling (larger grids = longer retention due to higher recompute cost)
        grid_factor = self.grid_multiplier
        base_msolve *= grid_factor
        base_mstep *= grid_factor
        base_morch *= grid_factor
        
        # Convert to ticks (scale by 1/dt_avg to get step counts, then by tick_rate)
        if self.dt_avg > 0:
            tick_factor = self.tick_rate / self.dt_avg
            msolve_ticks_s = int(base_msolve * tick_factor)
            msolve_ticks_l = int(base_msolve * tick_factor * 4)  # Long TTL = 4x short
            mstep_ticks_s = int(base_mstep * tick_factor)
            mstep_ticks_l = int(base_mstep * tick_factor * 4)
            morch_ticks_s = int(base_morch * tick_factor)
            morch_ticks_l = int(base_morch * tick_factor * 4)
        else:
            # Fallback if dt_avg is invalid
            msolve_ticks_s = self.MIN_MSOLVE_S
            msolve_ticks_l = self.MIN_MSOLVE_L
            mstep_ticks_s = self.MIN_MSTEP_S
            mstep_ticks_l = self.MIN_MSTEP_L
            morch_ticks_s = self.MIN_MORCH_S
            morch_ticks_l = self.MIN_MORCH_L
        
        # Apply min/max bounds for M_solve
        msolve_ticks_s = max(self.MIN_MSOLVE_S, min(msolve_ticks_s, self.MAX_MSOLVE_S))
        msolve_ticks_l = max(self.MIN_MSOLVE_L, min(msolve_ticks_l, self.MAX_MSOLVE_L))
        
        # Apply min/max bounds for M_step
        mstep_ticks_s = max(self.MIN_MSTEP_S, min(mstep_ticks_s, self.MAX_MSTEP_S))
        mstep_ticks_l = max(self.MIN_MSTEP_L, min(mstep_ticks_l, self.MAX_MSTEP_L))
        
        # Apply min/max bounds for M_orch
        morch_ticks_s = max(self.MIN_MORCH_S, min(morch_ticks_s, self.MAX_MORCH_S))
        morch_ticks_l = max(self.MIN_MORCH_L, min(morch_ticks_l, self.MAX_MORCH_L))
        
        logger.debug(f"Computed TTLs: M_solve=({msolve_ticks_s}/{msolve_ticks_l}), "
                     f"M_step=({mstep_ticks_s}/{mstep_ticks_l}), "
                     f"M_orch=({morch_ticks_s}/{morch_ticks_l})")
        
        return AdaptiveTTLs(
            msolve_ttl_s=msolve_ticks_s,
            msolve_ttl_l=msolve_ticks_l,
            mstep_ttl_s=mstep_ticks_s,
            mstep_ttl_l=mstep_ticks_l,
            morch_ttl_s=morch_ticks_s,
            morch_ttl_l=morch_ticks_l
        )
    
    def get_static_fallback(self) -> AdaptiveTTLs:
        """
        Return static TTL fallback values for backward compatibility.
        
        Returns:
            AdaptiveTTLs with static TTL values
        """
        return AdaptiveTTLs(
            msolve_ttl_s=self.MIN_MSOLVE_S,      # 1 hour
            msolve_ttl_l=self.MIN_MSOLVE_L,      # 1 day
            mstep_ttl_s=self.MIN_MSTEP_S,        # 10 hours
            mstep_ttl_l=self.MIN_MSTEP_L,        # 1 week
            morch_ttl_s=self.MIN_MORCH_S,        # 30 days
            morch_ttl_l=self.MIN_MORCH_L         # 30 days
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
