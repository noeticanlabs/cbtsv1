"""
Unified Clock Architecture for GR Solver

This module provides a shared clock state that both GRScheduler and MultiRateBandManager
reference, ensuring single source of truth for time state while maintaining modular
component design.

Key Classes:
- UnifiedClockState: Shared state with global_step, global_time, band_steps, band_times
- UnifiedClock: Main interface for clock operations used by scheduler and memory
- BandConfig: Configuration for per-band update cadences (octave-based)
"""

import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class UnifiedClockState:
    """
    Shared state for all clock operations.
    
    This is the single source of truth for time state in the GR solver.
    All components that need to track time reference this state.
    """
    global_step: int = 0
    global_time: float = 0.0
    band_steps: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.int64))
    band_times: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float64))
    
    def tick(self, dt: float):
        """Advance global time by dt."""
        self.global_step += 1
        self.global_time += dt
    
    def tick_to_step(self, step: int):
        """Advance global time to a specific step."""
        self.global_step = step
        self.global_time = step * getattr(self, '_base_dt', 0.001)
    
    def advance_bands(self, octave: int, interval: int):
        """Mark a band as updated at current step."""
        if octave < len(self.band_steps):
            if self.global_step % interval == 0:
                self.band_steps[octave] = self.global_step
                self.band_times[octave] = self.global_time
    
    def copy(self) -> 'UnifiedClockState':
        """Create a deep copy of the clock state."""
        new_state = UnifiedClockState(
            global_step=self.global_step,
            global_time=self.global_time,
            band_steps=self.band_steps.copy(),
            band_times=self.band_times.copy()
        )
        return new_state
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization/receipts."""
        return {
            'global_step': int(self.global_step),
            'global_time': float(self.global_time),
            'band_steps': self.band_steps.tolist(),
            'band_times': self.band_times.tolist()
        }


class BandConfig:
    """
    Configuration for per-band update cadences (octave-based).
    
    Higher octaves update less frequently according to power-of-2 cadence factors.
    This enables efficient band-selective updates where high-frequency bands
    are computed less often when signal is weak.
    """
    
    DEFAULT_OCTAVES = 8
    
    def __init__(self, base_dt: float, octaves: int = None):
        self.base_dt = base_dt
        self.octaves = octaves if octaves is not None else self.DEFAULT_OCTAVES
        # Cadence factors: octave 0 updates every step, octave 1 every 2 steps, etc.
        self.cadence_factors = [2 ** o for o in range(self.octaves)]
        self.band_thresholds = self._compute_band_thresholds()
    
    def _compute_band_thresholds(self) -> List[float]:
        """
        Compute amplitude thresholds for each octave band.
        
        Higher octaves have lower amplitude thresholds (more selective),
        meaning high-frequency bands require stronger signal to update.
        """
        thresholds = []
        for o in range(self.octaves):
            # Threshold decreases by factor of 2 per octave
            threshold = self.base_dt / (2 ** (o + 2))
            thresholds.append(threshold)
        return thresholds
    
    def get_update_interval(self, octave: int) -> int:
        """Get update interval in steps for a given octave."""
        return int(self.cadence_factors[min(octave, self.octaves - 1)])
    
    def get_threshold(self, octave: int) -> float:
        """Get amplitude threshold for culling a given octave."""
        return self.band_thresholds[min(octave, self.octaves - 1)]
    
    def get_cadence_factors(self) -> List[int]:
        """Get all cadence factors."""
        return self.cadence_factors.copy()
    
    def get_band_thresholds(self) -> List[float]:
        """Get all band thresholds."""
        return self.band_thresholds.copy()


class UnifiedClock:
    """
    Main clock interface for unified time management.
    
    This class provides the primary interface used by both GRScheduler
    and MultiRateBandManager. It manages:
    - Global time state (step, time)
    - Band-specific clock state (per-octave tracking)
    - DT constraint computation (CFL, gauge, coherence, resolution)
    - Regime detection and cache invalidation
    
    The unified clock ensures all components share the same time state,
    preventing desynchronization between scheduler and memory systems.
    """
    
    def __init__(self, base_dt: float = 0.001, octaves: int = None, 
                 receipt_emitter: Any = None):
        """
        Initialize the unified clock.
        
        Args:
            base_dt: Base timestep for clock operations
            octaves: Number of frequency bands (default: 8)
            receipt_emitter: Optional emitter for clock decision receipts
        """
        self.base_dt = base_dt
        self.octaves = octaves if octaves is not None else BandConfig.DEFAULT_OCTAVES
        self.receipt_emitter = receipt_emitter
        
        # Shared state - single source of truth
        self.state = UnifiedClockState()
        self.state._base_dt = base_dt  # Store for tick_to_step
        
        # Band configuration
        self.band_config = BandConfig(base_dt, self.octaves)
        
        # Regime tracking
        self.prev_dt: Optional[float] = None
        self.prev_residual_slope: Optional[float] = None
        self.prev_dominant_band: int = 0
        self.prev_resolution: int = 0
        self.current_regime_hash: Optional[str] = None
        self.regime_shift_detected: bool = False
        
        # Culling state
        self.culled_bands = np.zeros(self.octaves, dtype=bool)
        
        # Band metrics from external sources (e.g., phaseloom)
        self.dominant_band: int = 0
        self.amplitude: float = 0.0
    
    # =========================================================================
    # Global Clock Operations
    # =========================================================================
    
    def tick(self, dt: float = None) -> int:
        """
        Advance the global clock by one step.
        
        Args:
            dt: Time step (uses base_dt if not provided)
            
        Returns:
            Current global step number
        """
        step_dt = dt if dt is not None else self.base_dt
        self.state.tick(step_dt)
        
        # Advance per-band clocks based on cadence
        for o in range(self.octaves):
            interval = self.band_config.get_update_interval(o)
            self.state.advance_bands(o, interval)
        
        return self.state.global_step
    
    def tick_to(self, step: int, dt: float = None) -> int:
        """
        Advance the global clock to a specific step.
        
        Args:
            step: Target step number
            dt: Time step (uses base_dt if not provided)
            
        Returns:
            Current global step number
        """
        step_dt = dt if dt is not None else self.base_dt
        self.state._base_dt = step_dt
        self.state.tick_to_step(step)
        
        # Advance per-band clocks based on cadence
        for o in range(self.octaves):
            interval = self.band_config.get_update_interval(o)
            self.state.advance_bands(o, interval)
        
        return self.state.global_step
    
    def get_global_step(self) -> int:
        """Get current global step."""
        return self.state.global_step
    
    def get_global_time(self) -> float:
        """Get current global time."""
        return self.state.global_time
    
    def get_dt(self) -> float:
        """Get base dt."""
        return self.base_dt
    
    # =========================================================================
    # Band Clock Operations
    # =========================================================================
    
    def get_bands_to_update(self, dominant_band: int = 0, amplitude: float = 0.0) -> np.ndarray:
        """
        Determine which frequency bands should be updated.
        
        Uses octave culling: skip high-frequency bands when signal is weak.
        
        Args:
            dominant_band: Index of the dominant frequency band
            amplitude: Current signal amplitude for threshold comparisons
            
        Returns:
            Boolean array where True indicates band should be updated
        """
        update_mask = np.zeros(self.octaves, dtype=bool)
        self.regime_shift_detected = False
        
        for o in range(self.octaves):
            # Check if this band is due for update based on cadence
            interval = self.band_config.get_update_interval(o)
            is_due = (self.state.global_step % interval == 0)
            
            # Skip if not due yet (cadence-based culling)
            if not is_due:
                self.culled_bands[o] = True
                update_mask[o] = False
                continue
            
            # Check amplitude threshold (octave culling for high bands)
            # Higher octaves (o > dominant_band) need stronger signal
            if o > dominant_band:
                threshold = self.band_config.get_threshold(o)
                if amplitude < threshold:
                    self.culled_bands[o] = True
                    update_mask[o] = False
                    continue
            
            # Band is due and passes amplitude check
            self.culled_bands[o] = False
            update_mask[o] = True
        
        return update_mask
    
    def get_band_state(self, octave: int) -> Tuple[int, float]:
        """Get the step and time for a specific band."""
        if octave < len(self.state.band_steps):
            return int(self.state.band_steps[octave]), float(self.state.band_times[octave])
        return 0, 0.0
    
    def get_all_band_states(self) -> Dict[int, Tuple[int, float]]:
        """Get step and time for all bands."""
        return {
            o: (int(self.state.band_steps[o]), float(self.state.band_times[o]))
            for o in range(self.octaves)
        }
    
    # =========================================================================
    # DT Constraint Computation
    # =========================================================================
    
    def compute_dt_constraints(self, dt_candidate: float, fields, 
                                lambda_val: float = 0.0) -> Tuple[Dict, float]:
        """
        Compute all clock constraints and determine the timestep to use.
        
        This method consolidates constraint computation that was previously
        spread across GRScheduler and MultiRateClockSystem.
        
        Args:
            dt_candidate: Proposed timestep
            fields: GR fields object with alpha, beta, K_sym6, dx, dy, dz
            lambda_val: Constraint damping coefficient
            
        Returns:
            Tuple of (clocks_dict, dt_used)
        """
        # CFL constraint: dt < dx / c where c is characteristic speed
        c_max = np.sqrt(np.max(fields.alpha)**2 + np.max(np.linalg.norm(fields.beta, axis=-1))**2)
        h_min = min(fields.dx, fields.dy, fields.dz)
        dt_CFL = h_min / c_max if c_max > 0 else float('inf')
        
        # Gauge constraint: dt < alpha * h_min / (1 + |beta|)
        beta_norm = np.max(np.linalg.norm(fields.beta, axis=-1))
        dt_gauge = fields.alpha.max() * h_min / (1 + beta_norm)
        
        # Coherence (constraint damping): dt < 1 / lambda where lambda is damping rate
        dt_coh = 1.0 / max(lambda_val, 1e-6) if lambda_val > 0 else float('inf')
        
        # Resolution constraint: dt < h_min / sqrt(K^2)
        K_norm = np.max(np.linalg.norm(fields.K_sym6, axis=-1))
        dt_res = h_min / max(np.sqrt(K_norm), 1e-6)
        
        # Sigma (shock capturing or similar): placeholder
        dt_sigma = float('inf')
        
        # Choose minimum of all constraints
        dt_used = min(dt_candidate, dt_CFL, dt_gauge, dt_coh, dt_res, dt_sigma)
        
        clocks = {
            'dt_CFL': dt_CFL,
            'dt_gauge': dt_gauge,
            'dt_coh': dt_coh,
            'dt_res': dt_res,
            'dt_sigma': dt_sigma,
            'dt_used': dt_used,
            'multi_rate': {
                'cadence_factors': self.band_config.get_cadence_factors(),
                'band_thresholds': self.band_config.get_band_thresholds()
            }
        }
        
        return clocks, dt_used
    
    def compute_cfl_constraint(self, fields) -> float:
        """Compute just the CFL constraint."""
        c_max = np.sqrt(np.max(fields.alpha)**2 + np.max(np.linalg.norm(fields.beta, axis=-1))**2)
        h_min = min(fields.dx, fields.dy, fields.dz)
        return h_min / c_max if c_max > 0 else float('inf')
    
    def compute_gauge_constraint(self, fields) -> float:
        """Compute just the gauge constraint."""
        beta_norm = np.max(np.linalg.norm(fields.beta, axis=-1))
        h_min = min(fields.dx, fields.dy, fields.dz)
        return fields.alpha.max() * h_min / (1 + beta_norm)
    
    def compute_resolution_constraint(self, fields) -> float:
        """Compute just the resolution constraint."""
        h_min = min(fields.dx, fields.dy, fields.dz)
        K_norm = np.max(np.linalg.norm(fields.K_sym6, axis=-1))
        return h_min / max(np.sqrt(K_norm), 1e-6)
    
    # =========================================================================
    # Regime Detection and Cache Management
    # =========================================================================
    
    def compute_regime_hash(self, dt: float, dominant_band: int, resolution: int) -> str:
        """
        Compute regime hash based on dominant band, dt, and resolution.
        
        Regime shifts are detected when:
        - dt changes by more than 2x
        - residual slope sign changes
        
        Args:
            dt: Current timestep
            dominant_band: Dominant frequency band index
            resolution: Current grid resolution
            
        Returns:
            Regime hash string
        """
        # Check for regime shifts based on dt change
        if self.prev_dt is not None:
            dt_ratio = dt / self.prev_dt if self.prev_dt > 0 else float('inf')
            if dt_ratio > 2.0 or dt_ratio < 0.5:
                self.regime_shift_detected = True
        
        # Create regime hash input
        dt_bucket = int(np.log10(dt) * 10) if dt > 0 else -100
        hash_input = f"{dt_bucket}:{dominant_band}:{resolution}"
        
        self.current_regime_hash = hashlib.md5(hash_input.encode()).hexdigest()[:16]
        self.prev_dt = dt
        self.prev_dominant_band = dominant_band
        self.prev_resolution = resolution
        
        return self.current_regime_hash
    
    def detect_regime_shift(self, residual_slope: float) -> bool:
        """
        Detect regime shift from residual slope sign change.
        
        Args:
            residual_slope: Current residual slope
            
        Returns:
            True if regime shift detected
        """
        if self.prev_residual_slope is not None:
            sign_change = (residual_slope > 0) != (self.prev_residual_slope > 0)
            if sign_change and abs(residual_slope - self.prev_residual_slope) > 1e-6:
                self.regime_shift_detected = True
                self.prev_residual_slope = residual_slope
                return True
        
        self.prev_residual_slope = residual_slope
        return False
    
    def should_invalidate_cache(self) -> bool:
        """Check if cache should be invalidated due to regime shift."""
        return self.regime_shift_detected
    
    def reset_regime_shift(self):
        """Reset regime shift detection flag after handling."""
        self.regime_shift_detected = False
    
    def update_band_metrics(self, dominant_band: int, amplitude: float):
        """
        Update band metrics from external source (e.g., phaseloom computation).
        
        Args:
            dominant_band: Index of the dominant frequency band
            amplitude: Current signal amplitude for threshold comparisons
        """
        self.dominant_band = dominant_band
        self.amplitude = amplitude
    
    # =========================================================================
    # Summary and Diagnostics
    # =========================================================================
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of clock state for diagnostics."""
        return {
            'global_step': int(self.state.global_step),
            'global_time': float(self.state.global_time),
            'band_steps': self.state.band_steps.tolist(),
            'band_times': self.state.band_times.tolist(),
            'regime_hash': self.current_regime_hash,
            'regime_shift': self.regime_shift_detected,
            'culled_bands': self.culled_bands.tolist()
        }
    
    def emit_clock_decision_receipt(self, step: int, t: float, dt: float, clocks: Dict):
        """Emit receipt for clock decision."""
        if hasattr(self.receipt_emitter, 'emit_clock_decision_receipt'):
            self.receipt_emitter.emit_clock_decision_receipt(step, t, dt, clocks)
    
    # =========================================================================
    # State Management
    # =========================================================================
    
    def get_state(self) -> UnifiedClockState:
        """Get the current clock state."""
        return self.state
    
    def set_state(self, state: UnifiedClockState):
        """Set the clock state (for rollback/restoration)."""
        self.state = state.copy()
        if hasattr(state, '_base_dt'):
            self.base_dt = state._base_dt
    
    def snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of the clock state for serialization."""
        return {
            'state': self.state.to_dict(),
            'prev_dt': self.prev_dt,
            'prev_residual_slope': self.prev_residual_slope,
            'prev_dominant_band': self.prev_dominant_band,
            'prev_resolution': self.prev_resolution,
            'current_regime_hash': self.current_regime_hash,
            'regime_shift_detected': self.regime_shift_detected,
            'dominant_band': self.dominant_band,
            'amplitude': self.amplitude
        }
    
    def restore(self, snapshot: Dict[str, Any]):
        """Restore clock state from a snapshot."""
        state_dict = snapshot['state']
        self.state = UnifiedClockState(
            global_step=state_dict['global_step'],
            global_time=state_dict['global_time'],
            band_steps=np.array(state_dict['band_steps'], dtype=np.int64),
            band_times=np.array(state_dict['band_times'], dtype=np.float64)
        )
        self.prev_dt = snapshot['prev_dt']
        self.prev_residual_slope = snapshot['prev_residual_slope']
        self.prev_dominant_band = snapshot['prev_dominant_band']
        self.prev_resolution = snapshot['prev_resolution']
        self.current_regime_hash = snapshot['current_regime_hash']
        self.regime_shift_detected = snapshot['regime_shift_detected']
        self.dominant_band = snapshot['dominant_band']
        self.amplitude = snapshot['amplitude']


# Aliases for backward compatibility
ClockState = UnifiedClockState
BandClockConfig = BandConfig
