import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple
from .gr_receipts import ReceiptEmitter
from .gr_clock import UnifiedClock, BandConfig, UnifiedClockState


class BandClockConfig:
    """Configuration for per-band update cadences (octave-based)."""
    
    def __init__(self, base_dt: float, octaves: int = 8):
        self.base_dt = base_dt
        self.octaves = octaves
        # Cadence factors: higher octaves update less frequently
        # octave 0: every step, octave 1: every 2 steps, ..., octave 7: every 128 steps
        self.cadence_factors = [2 ** o for o in range(octaves)]
        self.band_thresholds = self._compute_band_thresholds()
    
    def _compute_band_thresholds(self) -> List[float]:
        """Compute amplitude thresholds for each octave band."""
        # Higher octaves have lower amplitude thresholds (more selective)
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


class MultiRateBandManager:
    """
    Band manager for multi-rate updates with unified clock support.
    
    This class manages band-specific logic (cadence, thresholds, regime detection)
    while using UnifiedClock for shared state. It replaces MultiRateClockSystem
    for new code but maintains backward compatibility.
    
    Args:
        unified_clock: UnifiedClock instance for shared time state
        receipt_emitter: Optional receipt emitter for logging
    """
    
    DEFAULT_OCTAVES = 8
    
    def __init__(self, unified_clock: UnifiedClock = None, receipt_emitter: ReceiptEmitter = None,
                 base_dt: float = 0.001, octaves: int = None):
        """
        Initialize the band manager.
        
        Can be initialized with either a UnifiedClock (new style) or
        with separate parameters (legacy style for backward compatibility).
        """
        self.receipt_emitter = receipt_emitter
        self.octaves = octaves if octaves is not None else self.DEFAULT_OCTAVES
        
        if unified_clock is not None:
            # New style: use provided unified clock
            self._unified_clock = unified_clock
            self._owns_clock = False
            self.band_config = unified_clock.band_config
        else:
            # Legacy style: create internal clock for backward compatibility
            self._unified_clock = UnifiedClock(base_dt=base_dt, octaves=self.octaves, 
                                               receipt_emitter=receipt_emitter)
            self._owns_clock = True
            self.band_config = BandConfig(base_dt, self.octaves)
    
    @property
    def unified_clock(self) -> UnifiedClock:
        """Get the unified clock."""
        return self._unified_clock
    
    @property
    def global_step(self) -> int:
        """Get global step from unified clock."""
        return self._unified_clock.get_global_step()
    
    @global_step.setter
    def global_step(self, value: int):
        """Set global step in unified clock."""
        self._unified_clock.state.global_step = value
    
    @property
    def global_time(self) -> float:
        """Get global time from unified clock."""
        return self._unified_clock.get_global_time()
    
    @global_time.setter
    def global_time(self, value: float):
        """Set global time in unified clock."""
        self._unified_clock.state.global_time = value
    
    @property
    def band_steps(self) -> np.ndarray:
        """Get band steps from unified clock."""
        return self._unified_clock.state.band_steps
    
    @property
    def band_times(self) -> np.ndarray:
        """Get band times from unified clock."""
        return self._unified_clock.state.band_times
    
    @property
    def current_regime_hash(self) -> Optional[str]:
        """Get current regime hash."""
        return self._unified_clock.current_regime_hash
    
    @property
    def regime_shift_detected(self) -> bool:
        """Check if regime shift was detected."""
        return self._unified_clock.regime_shift_detected
    
    def tick_global(self, dt: float, step: int = None):
        """Advance global clock by one step or to specified step."""
        if step is not None:
            self._unified_clock.tick_to(step, dt)
        else:
            self._unified_clock.tick(dt)
    
    def compute_clocks(self, dt_candidate: float, fields) -> Tuple[Dict, float]:
        """Compute CFL-aware clocks with multi-rate support (delegates to unified clock)."""
        return self._unified_clock.compute_dt_constraints(dt_candidate, fields)
    
    def get_bands_to_update(self, dominant_band: int = 0, amplitude: float = 0.0) -> np.ndarray:
        """Determine which frequency bands should be updated."""
        return self._unified_clock.get_bands_to_update(dominant_band, amplitude)
    
    def compute_regime_hash(self, dt: float, dominant_band: int, resolution: int) -> str:
        """Compute regime hash based on dominant band, dt, and resolution."""
        return self._unified_clock.compute_regime_hash(dt, dominant_band, resolution)
    
    def detect_regime_shift(self, residual_slope: float) -> bool:
        """Detect regime shift from residual slope sign change."""
        return self._unified_clock.detect_regime_shift(residual_slope)
    
    def should_invalidate_cache(self) -> bool:
        """Check if cache should be invalidated due to regime shift."""
        return self._unified_clock.should_invalidate_cache()
    
    def reset_regime_shift(self):
        """Reset regime shift detection flag after handling."""
        self._unified_clock.reset_regime_shift()
    
    def get_multi_rate_summary(self) -> Dict:
        """Get summary of multi-rate clock state for diagnostics."""
        return self._unified_clock.get_summary()
    
    def emit_clock_decision_receipt(self, step: int, t: float, dt: float, clocks: Dict):
        """Emit receipt for clock decision (delegates to receipt emitter)."""
        if hasattr(self.receipt_emitter, 'emit_clock_decision_receipt'):
            self.receipt_emitter.emit_clock_decision_receipt(step, t, dt, clocks)
    
    def update_band_metrics(self, dominant_band: int, amplitude: float):
        """Update band metrics from external source."""
        self._unified_clock.update_band_metrics(dominant_band, amplitude)
    
    def get_summary(self) -> Dict:
        """Get summary of band manager state."""
        return self._unified_clock.get_summary()


class MultiRateClockSystem:
    """
    Multi-rate scheduler for band-selective updates.
    
    .. admonition:: Legacy
       :class: note
       
       Use MultiRateBandManager with UnifiedClock instead for new code.
       This class is maintained for backward compatibility.
    
    Manages clock hierarchy for HPC integration with:
    - Band-specific update cadences (octave-based)
    - Regime hash computation
    - Cache invalidation on regime shifts
    """
    
    # FIX: Changed from 27 to 8 to match phaseloom_octaves.py max_octaves=8
    # The 27 value was for full phaseloom, but we only use 8 octaves for band-selective updates
    DEFAULT_OCTAVES = 8
    
    def __init__(self, receipt_emitter: ReceiptEmitter, base_dt: float = 0.001, octaves: int = None):
        """
        Initialize the multi-rate clock system.
        
        For new code, consider using MultiRateBandManager with UnifiedClock instead.
        """
        self.receipt_emitter = receipt_emitter
        self.base_dt = base_dt
        self.octaves = octaves if octaves is not None else self.DEFAULT_OCTAVES
        self.band_config = BandClockConfig(base_dt, self.octaves)
        
        # Multi-rate clock state
        self.global_step = 0
        self.global_time = 0.0
        
        # Per-band clock state: tracks when each band was last updated
        self.band_steps = np.zeros(self.octaves, dtype=np.int64)
        self.band_times = np.zeros(self.octaves, dtype=np.float64)
        
        # Regime tracking
        self.prev_dt: Optional[float] = None
        self.prev_residual_slope: Optional[float] = None
        self.prev_dominant_band: int = 0
        self.prev_resolution: int = 0
        self.current_regime_hash: Optional[str] = None
        self.regime_shift_detected: bool = False
        
        # Culling state
        self.culled_bands = np.zeros(self.octaves, dtype=bool)
    
    def compute_clocks(self, dt_candidate: float, fields) -> Tuple[Dict, float]:
        """Compute CFL-aware clocks with multi-rate support."""
        c_max = np.sqrt(np.max(fields.alpha)**2 + np.max(np.linalg.norm(fields.beta, axis=-1))**2)
        h_min = min(fields.dx, fields.dy, fields.dz)
        dt_CFL = h_min / c_max if c_max > 0 else float('inf')
        
        beta_norm = np.max(np.linalg.norm(fields.beta, axis=-1))
        dt_gauge = fields.alpha.max() * h_min / (1 + beta_norm) if fields.alpha.size > 0 else float('inf')
        
        dt_coh = dt_CFL * 0.5
        
        K_norm = np.max(np.linalg.norm(fields.K_sym6, axis=-1))
        dt_res = h_min / max(np.sqrt(K_norm), 1e-6)
        
        dt_sigma = float('inf')
        dt_used = min(dt_candidate, dt_CFL, dt_gauge, dt_coh, dt_res, dt_sigma)
        
        clocks = {
            'dt_CFL': dt_CFL,
            'dt_gauge': dt_gauge,
            'dt_coh': dt_coh,
            'dt_res': dt_res,
            'dt_sigma': dt_sigma,
            'dt_used': dt_used,
            'multi_rate': {
                'cadence_factors': self.band_config.cadence_factors,
                'band_thresholds': self.band_config.band_thresholds
            }
        }
        
        return clocks, dt_used
    
    def tick_global(self, dt: float, step: int = None):
        """Advance global clock by one step or to specified step."""
        if step is not None:
            self.global_step = step
        else:
            self.global_step += 1
        self.global_time += dt
        
        # Advance per-band clocks based on cadence
        for o in range(self.octaves):
            interval = self.band_config.get_update_interval(o)
            if self.global_step % interval == 0:
                self.band_steps[o] = self.global_step
                self.band_times[o] = self.global_time
    
    def get_bands_to_update(self, dominant_band: int, amplitude: float) -> np.ndarray:
        """
        Determine which frequency bands should be updated.
        
        Returns boolean array where True indicates band should be updated.
        Uses octave culling: skip high-frequency bands when signal is weak.
        """
        update_mask = np.zeros(self.octaves, dtype=bool)
        self.regime_shift_detected = False
        
        for o in range(self.octaves):
            # Check if this band is due for update based on cadence
            interval = self.band_config.get_update_interval(o)
            is_due = (self.global_step % interval == 0)
            
            # Skip if not due yet (cadence-based culling)
            if not is_due:
                self.culled_bands[o] = True
                update_mask[o] = False
                continue
            
            # Check amplitude threshold (octave culling for high bands)
            # Higher octaves (o > dominant_band) need stronger signal
            if o > dominant_band:
                threshold = self.band_config.get_threshold(o)
                # Amplitude must be significant to update high octaves
                if amplitude < threshold:
                    self.culled_bands[o] = True
                    update_mask[o] = False
                    continue
            
            # Band is due and passes amplitude check
            self.culled_bands[o] = False
            update_mask[o] = True
        
        return update_mask
    
    def compute_regime_hash(self, dt: float, dominant_band: int, resolution: int) -> str:
        """
        Compute regime hash based on dominant band, dt, and resolution.
        
        Regime shifts are detected when:
        - dt changes by more than 2x
        - residual slope sign changes
        """
        # Check for regime shifts
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
        """Detect regime shift from residual slope sign change."""
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
    
    def get_multi_rate_summary(self) -> Dict:
        """Get summary of multi-rate clock state for diagnostics."""
        return {
            'global_step': int(self.global_step),
            'global_time': float(self.global_time),
            'band_steps': self.band_steps.tolist(),
            'band_times': self.band_times.tolist(),
            'regime_hash': self.current_regime_hash,
            'regime_shift': self.regime_shift_detected,
            'culled_bands': self.culled_bands.tolist()
        }
    
    def emit_clock_decision_receipt(self, step: int, t: float, dt: float, clocks: Dict):
        """Emit receipt for clock decision (delegates to receipt emitter)."""
        if hasattr(self.receipt_emitter, 'emit_clock_decision_receipt'):
            self.receipt_emitter.emit_clock_decision_receipt(step, t, dt, clocks)
    
    def update_band_metrics(self, dominant_band: int, amplitude: float):
        """
        Update band metrics from external source (e.g., phaseloom computation).
        
        Args:
            dominant_band: Index of the dominant frequency band
            amplitude: Current signal amplitude for threshold comparisons
        """
        self.dominant_band = dominant_band
        self.amplitude = amplitude


class ClockManager:
    """Legacy ClockManager for backward compatibility."""
    
    def __init__(self, receipt_emitter: ReceiptEmitter):
        self.receipt_emitter = receipt_emitter

    def compute_clocks(self, dt_candidate, fields):
        c_max = np.sqrt(np.max(fields.alpha)**2 + np.max(np.linalg.norm(fields.beta, axis=-1))**2)
        h_min = min(fields.dx, fields.dy, fields.dz)
        dt_CFL = h_min / c_max if c_max > 0 else float('inf')

        beta_norm = np.max(np.linalg.norm(fields.beta, axis=-1))
        dt_gauge = fields.alpha.max() * h_min / (1 + beta_norm) if fields.alpha.size > 0 else float('inf')

        dt_coh = dt_CFL * 0.5

        K_norm = np.max(np.linalg.norm(fields.K_sym6, axis=-1))
        dt_res = h_min / max(np.sqrt(K_norm), 1e-6)

        dt_sigma = float('inf')

        dt_used = min(dt_candidate, dt_CFL, dt_gauge, dt_coh, dt_res, dt_sigma)

        clocks = {
            'dt_CFL': dt_CFL,
            'dt_gauge': dt_gauge,
            'dt_coh': dt_coh,
            'dt_res': dt_res,
            'dt_sigma': dt_sigma,
            'dt_used': dt_used
        }

        return clocks, dt_used

    def emit_clock_decision_receipt(self, step, t, dt, clocks):
        if hasattr(self.receipt_emitter, 'emit_clock_decision_receipt'):
            self.receipt_emitter.emit_clock_decision_receipt(step, t, dt, clocks)
