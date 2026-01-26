import numpy as np
from typing import Dict, Optional, Tuple
from .gr_clocks import MultiRateClockSystem, MultiRateBandManager
from .gr_clock import UnifiedClock


class PhaseLoomMemory:
    """
    Memory and state tracking for PhaseLoom operations.
    
    Integrates with UnifiedClock or MultiRateClockSystem for band-selective updates.
    Tracks which frequency bands were updated and implements octave culling.
    """
    
    def __init__(self, fields, clock_system: Optional = None, base_dt: float = 0.001,
                 unified_clock: Optional[UnifiedClock] = None):
        """
        Initialize PhaseLoomMemory.
        
        Args:
            fields: GR fields object
            clock_system: Legacy MultiRateClockSystem (for backward compatibility)
            base_dt: Base timestep
            unified_clock: UnifiedClock for shared time state (new style)
        """
        self.fields = fields
        self.step_count = 0
        self.last_loom_step = 0
        self.prev_K = None
        self.prev_gamma = None
        self.prev_residual_slope = None
        self.summary = {}
        self.D_max_prev = 0.0
        self.tainted = False
        
        # Clock system - support both legacy and new style
        if unified_clock is not None:
            self._unified_clock = unified_clock
            self.clock_system = None  # Use unified_clock directly
        elif clock_system is not None:
            self.clock_system = clock_system
            self._unified_clock = None
        else:
            self.clock_system = None
            self._unified_clock = None
        
        self.base_dt = base_dt
        
        # Band tracking
        self.bands_updated: np.ndarray = np.zeros(8, dtype=bool)  # 8 octaves default
        self.dominant_band = 0
        self.amplitude = 0.0
        
        # Octave culling thresholds
        self.octave_cull_threshold = 1e-6
    
    @property
    def unified_clock(self) -> Optional[UnifiedClock]:
        """Get the unified clock if available."""
        if self._unified_clock is not None:
            return self._unified_clock
        if self.clock_system is not None and hasattr(self.clock_system, '_unified_clock'):
            return self.clock_system._unified_clock
        return None
    
    def initialize_clock_system(self, receipt_emitter) -> MultiRateClockSystem:
        """Initialize the legacy multi-rate clock system if not provided."""
        if self.clock_system is None and self._unified_clock is None:
            self.clock_system = MultiRateClockSystem(receipt_emitter, self.base_dt)
        return self.clock_system
    
    def set_unified_clock(self, clock: UnifiedClock):
        """Set the unified clock for shared time state."""
        self._unified_clock = clock
        self.clock_system = None  # Prefer unified clock over legacy
    
    def should_compute_loom(self, step: int, K: np.ndarray, gamma: np.ndarray, 
                           residual_slope: float, rollback_occurred: bool,
                           amplitude: float = 0.0, dominant_band: int = 0,
                           resolution: int = 64) -> Tuple[bool, np.ndarray]:
        """
        Determine if loom computation is needed using clock-based triggering.
        
        Args:
            step: Current global step
            K: Curvature tensor state
            gamma: Connection coefficient state
            residual_slope: Current residual slope
            rollback_occurred: Whether a rollback occurred
            amplitude: Current signal amplitude (for octave culling)
            dominant_band: Dominant frequency band index
            resolution: Current grid resolution
            
        Returns:
            Tuple of (should_compute, bands_to_update)
        """
        self.step_count = step
        self.amplitude = amplitude
        self.dominant_band = dominant_band
        
        # Compute delta metrics
        delta_K = np.max(np.abs(K - self.prev_K)) if self.prev_K is not None else 0.0
        delta_gamma = np.max(np.abs(gamma - self.prev_gamma)) if self.prev_gamma is not None else 0.0
        
        # Use unified clock if available (new style)
        if self._unified_clock is not None:
            return self._should_compute_with_unified_clock(
                step, delta_K, delta_gamma, residual_slope, rollback_occurred,
                amplitude, dominant_band, resolution
            )
        
        # Use legacy clock system
        if self.clock_system is not None:
            return self._should_compute_with_legacy_clock(
                step, delta_K, delta_gamma, residual_slope, rollback_occurred,
                amplitude, dominant_band, resolution
            )
        
        # Fallback to original hardcoded logic
        return self._legacy_should_compute_loom(step, delta_K, delta_gamma, residual_slope, rollback_occurred)
    
    def _should_compute_with_unified_clock(self, step: int, delta_K: float, delta_gamma: float,
                                           residual_slope: float, rollback_occurred: bool,
                                           amplitude: float, dominant_band: int,
                                           resolution: int) -> Tuple[bool, np.ndarray]:
        """Compute loom triggering using unified clock."""
        # Update unified clock
        self._unified_clock.tick(self.base_dt)
        self._unified_clock.state.global_step = step
        
        # Check for regime shifts
        regime_shift = self._unified_clock.detect_regime_shift(residual_slope)
        if regime_shift:
            self.tainted = True
        
        # Compute regime hash
        self._unified_clock.compute_regime_hash(self.base_dt, dominant_band, resolution)
        
        # Get bands to update based on multi-rate scheduling
        bands_to_update = self._unified_clock.get_bands_to_update(dominant_band, amplitude)
        self.bands_updated = bands_to_update
        
        # Force loom computation if regime shift detected
        if self._unified_clock.should_invalidate_cache():
            self.tainted = True
            return True, bands_to_update
        
        # Check if any bands need updating
        if not np.any(bands_to_update):
            return False, bands_to_update
        
        # Check delta thresholds for non-dominant bands (octave culling)
        for o in range(len(bands_to_update)):
            if bands_to_update[o] and o > dominant_band:
                threshold = self.octave_cull_threshold * (2 ** (dominant_band - o))
                if delta_K < threshold and delta_gamma < threshold:
                    bands_to_update[o] = False
        
        return np.any(bands_to_update), bands_to_update
    
    def _should_compute_with_legacy_clock(self, step: int, delta_K: float, delta_gamma: float,
                                          residual_slope: float, rollback_occurred: bool,
                                          amplitude: float, dominant_band: int,
                                          resolution: int) -> Tuple[bool, np.ndarray]:
        """Compute loom triggering using legacy clock system."""
        # Update clock system
        self.clock_system.tick_global(self.base_dt, step)
        
        # Check for regime shifts
        regime_shift = self.clock_system.detect_regime_shift(residual_slope)
        if regime_shift:
            self.tainted = True
        
        # Compute regime hash
        self.clock_system.compute_regime_hash(self.base_dt, dominant_band, resolution)
        
        # Get bands to update based on multi-rate scheduling
        bands_to_update = self.clock_system.get_bands_to_update(dominant_band, amplitude)
        self.bands_updated = bands_to_update
        
        # Force loom computation if regime shift detected
        if self.clock_system.should_invalidate_cache():
            self.tainted = True
            return True, bands_to_update
        
        # Check if any bands need updating
        if not np.any(bands_to_update):
            return False, bands_to_update
        
        # Check delta thresholds for non-dominant bands (octave culling)
        for o in range(len(bands_to_update)):
            if bands_to_update[o] and o > dominant_band:
                threshold = self.octave_cull_threshold * (2 ** (dominant_band - o))
                if delta_K < threshold and delta_gamma < threshold:
                    bands_to_update[o] = False
        
        return np.any(bands_to_update), bands_to_update
    
    def _legacy_should_compute_loom(self, step: int, delta_K: float, delta_gamma: float,
                                    residual_slope: float, rollback_occurred: bool) -> Tuple[bool, np.ndarray]:
        """Legacy triggering logic for backward compatibility."""
        if step - self.last_loom_step >= 4:
            # Cheap proxy gate: skip FFT if changes are tiny
            if delta_K < 1e-4 and delta_gamma < 1e-4 and abs(residual_slope) < 1e-5:
                return False, np.zeros(8, dtype=bool)
            return True, np.ones(8, dtype=bool)
        if delta_K > 1e-3:
            return True, np.ones(8, dtype=bool)
        if delta_gamma > 1e-3:
            return True, np.ones(8, dtype=bool)
        if abs(residual_slope) > 1e-5:
            return True, np.ones(8, dtype=bool)
        if rollback_occurred:
            return True, np.ones(8, dtype=bool)
        return False, np.zeros(8, dtype=bool)
    
    def post_loom_update(self, loom_data: Dict, step: int) -> Tuple[int, float]:
        """
        Update state after loom computation.
        
        Args:
            loom_data: Data from loom computation including band info
            step: Current step number
            
        Returns:
            Tuple of (dominant_band, amplitude) for clock system update
        """
        self.summary = loom_data
        self.D_max_prev = loom_data.get('D_max', 0.0)
        self.last_loom_step = step
        
        # Extract band metrics from loom data
        dominant_band = loom_data.get('dominant_band', 0)
        amplitude = loom_data.get('amplitude', 0.0)
        
        # Update local state
        self.dominant_band = dominant_band
        self.amplitude = amplitude
        
        # Update previous state for delta computation
        if 'K' in loom_data:
            self.prev_K = loom_data['K'].copy() if hasattr(loom_data['K'], 'copy') else loom_data['K']
        if 'gamma' in loom_data:
            self.prev_gamma = loom_data['gamma'].copy() if hasattr(loom_data['gamma'], 'copy') else loom_data['gamma']
            
        # Update unified clock if available
        if self._unified_clock is not None:
            self._unified_clock.reset_regime_shift()
            self._unified_clock.update_band_metrics(dominant_band, amplitude)
        
        # Update legacy clock system if it's a MultiRateClockSystem/MultiRateBandManager
        if self.clock_system is not None and hasattr(self.clock_system, 'reset_regime_shift'):
            self.clock_system.reset_regime_shift()
            if hasattr(self.clock_system, 'update_band_metrics'):
                self.clock_system.update_band_metrics(dominant_band, amplitude)
        
        return dominant_band, amplitude
    
    def get_bands_to_update(self) -> np.ndarray:
        """Get the mask of bands that were updated in last computation."""
        return self.bands_updated.copy()
    
    def get_dominant_band(self) -> int:
        """Get the current dominant band index."""
        return self.dominant_band
    
    def get_clock_summary(self) -> Dict:
        """Get summary of clock system state for diagnostics."""
        if self._unified_clock is not None:
            return self._unified_clock.get_summary()
        if self.clock_system is not None:
            return self.clock_system.get_multi_rate_summary()
        return {
            'global_step': int(self.step_count),
            'global_time': float(self.step_count * self.base_dt),
            'band_steps': [0] * 8,
            'band_times': [0.0] * 8,
            'regime_hash': None,
            'regime_shift': False,
            'culled_bands': [False] * 8
        }
    
    def honesty_check(self, skipped: bool) -> bool:
        """Check if skipping loom computation was honest (within tolerance)."""
        if skipped and self.D_max_prev > 1e-2:
            return True
        return False
    
    def set_clock_system(self, clock_system: MultiRateClockSystem):
        """Set the legacy multi-rate clock system for triggering."""
        self.clock_system = clock_system
        self._unified_clock = None  # Prefer legacy when explicitly set
    
    def get_regime_hash(self) -> Optional[str]:
        """Get current regime hash if clock system is available."""
        if self._unified_clock is not None:
            return self._unified_clock.current_regime_hash
        if self.clock_system is not None:
            return self.clock_system.current_regime_hash
        return None
    
    def check_regime_shift(self) -> bool:
        """Check if a regime shift was detected."""
        if self._unified_clock is not None:
            return self._unified_clock.regime_shift_detected
        if self.clock_system is not None:
            return self.clock_system.regime_shift_detected
        return False
    
    def update_clock_system_with_band_metrics(self, dominant_band: int, amplitude: float):
        """
        Pass computed band metrics to the clock system.
        
        Args:
            dominant_band: Dominant frequency band index from phaseloom analysis
            amplitude: Signal amplitude from phaseloom analysis
        """
        if self.clock_system is not None and hasattr(self.clock_system, 'update_band_metrics'):
            self.clock_system.update_band_metrics(dominant_band, amplitude)
            self.dominant_band = dominant_band
            self.amplitude = amplitude
