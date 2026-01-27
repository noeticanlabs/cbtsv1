"""Tests for multi-rate clock system and band-selective updates."""
import numpy as np
import pytest
from unittest.mock import MagicMock


class TestBandClockConfig:
    """Tests for BandClockConfig class."""
    
    def test_cadence_factors(self):
        """Test that cadence factors are powers of 2."""
        from src.core.gr_clocks import BandClockConfig
        
        config = BandClockConfig(base_dt=0.001, octaves=8)
        
        # Cadence factors should be 2^0, 2^1, ..., 2^7
        expected = [2 ** o for o in range(8)]
        assert config.cadence_factors == expected
    
    def test_update_intervals(self):
        """Test that update intervals match cadence factors."""
        from src.core.gr_clocks import BandClockConfig
        
        config = BandClockConfig(base_dt=0.001, octaves=4)
        
        assert config.get_update_interval(0) == 1
        assert config.get_update_interval(1) == 2
        assert config.get_update_interval(2) == 4
        assert config.get_update_interval(3) == 8
    
    def test_band_thresholds(self):
        """Test that band thresholds decrease with octave."""
        from src.core.gr_clocks import BandClockConfig
        
        config = BandClockConfig(base_dt=0.001, octaves=4)
        thresholds = config.band_thresholds
        
        # Higher octaves should have lower thresholds
        for o in range(1, len(thresholds)):
            assert thresholds[o] < thresholds[o - 1]


class TestMultiRateClockSystem:
    """Tests for MultiRateClockSystem class."""
    
    def test_initialization(self):
        """Test clock system initialization."""
        from src.core.gr_clocks import MultiRateClockSystem
        
        mock_emitter = MagicMock()
        clock_system = MultiRateClockSystem(mock_emitter, base_dt=0.001, octaves=4)
        
        assert clock_system.global_step == 0
        assert clock_system.global_time == 0.0
        assert clock_system.octaves == 4
        assert clock_system.current_regime_hash is None
    
    def test_tick_global(self):
        """Test global clock ticking."""
        from src.core.gr_clocks import MultiRateClockSystem
        
        mock_emitter = MagicMock()
        clock_system = MultiRateClockSystem(mock_emitter, base_dt=0.001, octaves=4)
        
        clock_system.tick_global(0.001)
        
        assert clock_system.global_step == 1
        assert clock_system.global_time == 0.001
    
    def test_get_bands_to_update_cadence(self):
        """Test that bands are updated based on cadence."""
        from src.core.gr_clocks import MultiRateClockSystem
        
        mock_emitter = MagicMock()
        clock_system = MultiRateClockSystem(mock_emitter, base_dt=0.001, octaves=4)
        
        # Step 1: Only band 0 should be due (interval 1)
        clock_system.global_step = 1
        bands = clock_system.get_bands_to_update(0, 1.0)
        
        # Only octave 0 is due at step 1
        assert bands[0] == True
        assert bands[1] == False
        assert bands[2] == False
        assert bands[3] == False
        
        # Step 2: Bands 0 and 1 should be due (intervals 1 and 2)
        clock_system.global_step = 2
        bands = clock_system.get_bands_to_update(0, 1.0)
        
        assert bands[0] == True
        assert bands[1] == True
        assert bands[2] == False
        assert bands[3] == False
        
        # Step 8: All bands should be due (intervals 1, 2, 4, 8 all divide 8)
        clock_system.global_step = 8
        bands = clock_system.get_bands_to_update(0, 1.0)
        
        assert np.all(bands)
    
    def test_get_bands_to_update_culling(self):
        """Test octave culling for low amplitude."""
        from src.core.gr_clocks import MultiRateClockSystem
        
        mock_emitter = MagicMock()
        clock_system = MultiRateClockSystem(mock_emitter, base_dt=0.001, octaves=4)
        
        # Step 8: All bands due (multiples of 8)
        clock_system.global_step = 8
        
        # Very low amplitude should cull higher octaves
        bands = clock_system.get_bands_to_update(1, 1e-10)
        
        # Higher octaves (3+) should be culled
        assert not bands[3]  # octave 3: threshold is very low
    
    def test_regime_hash_computation(self):
        """Test regime hash computation."""
        from src.core.gr_clocks import MultiRateClockSystem
        
        mock_emitter = MagicMock()
        clock_system = MultiRateClockSystem(mock_emitter, base_dt=0.001, octaves=4)
        
        hash1 = clock_system.compute_regime_hash(0.001, 0, 64)
        hash2 = clock_system.compute_regime_hash(0.001, 0, 64)
        hash3 = clock_system.compute_regime_hash(0.002, 0, 64)  # Different dt
        
        # Same inputs should give same hash
        assert hash1 == hash2
        # Different inputs should give different hash
        assert hash1 != hash3
    
    def test_dt_change_regime_shift(self):
        """Test regime shift detection on dt change > 2x."""
        from src.core.gr_clocks import MultiRateClockSystem
        
        mock_emitter = MagicMock()
        clock_system = MultiRateClockSystem(mock_emitter, base_dt=0.001, octaves=4)
        
        # First computation establishes baseline
        clock_system.compute_regime_hash(0.001, 0, 64)
        
        # dt changes by factor of 3 (> 2x) should trigger regime shift
        clock_system.compute_regime_hash(0.003, 0, 64)
        
        assert clock_system.regime_shift_detected
    
    def test_residual_slope_regime_shift(self):
        """Test regime shift detection on residual slope sign change."""
        from src.core.gr_clocks import MultiRateClockSystem
        
        mock_emitter = MagicMock()
        clock_system = MultiRateClockSystem(mock_emitter, base_dt=0.001, octaves=4)
        
        # First slope is positive
        clock_system.detect_regime_shift(0.1)
        assert not clock_system.regime_shift_detected
        
        # Sign change to negative should trigger
        slope_changed = clock_system.detect_regime_shift(-0.1)
        
        assert slope_changed
    
    def test_cache_invalidation(self):
        """Test that cache is invalidated on regime shift."""
        from src.core.gr_clocks import MultiRateClockSystem
        
        mock_emitter = MagicMock()
        clock_system = MultiRateClockSystem(mock_emitter, base_dt=0.001, octaves=4)
        
        # No regime shift initially
        assert not clock_system.regime_shift_detected
        
        # Establish baseline (first call sets prev_dt)
        clock_system.compute_regime_hash(0.001, 0, 64)
        assert not clock_system.regime_shift_detected
        
        # Trigger regime shift via dt change > 2x
        clock_system.compute_regime_hash(0.003, 0, 64)  # dt > 2x
        
        assert clock_system.regime_shift_detected
        
        # Reset after handling
        clock_system.reset_regime_shift()
        assert not clock_system.regime_shift_detected
    
    def test_multi_rate_summary(self):
        """Test multi-rate summary generation."""
        from src.core.gr_clocks import MultiRateClockSystem
        
        mock_emitter = MagicMock()
        clock_system = MultiRateClockSystem(mock_emitter, base_dt=0.001, octaves=4)
        
        clock_system.global_step = 10
        clock_system.global_time = 0.01
        
        summary = clock_system.get_multi_rate_summary()
        
        assert summary['global_step'] == 10
        assert summary['global_time'] == 0.01
        assert 'regime_hash' in summary
        assert 'culled_bands' in summary


class TestPhaseLoomMemoryMultiRate:
    """Tests for PhaseLoomMemory with multi-rate clock integration."""
    
    def test_initialization(self):
        """Test PhaseLoomMemory initialization."""
        from src.phaseloom.phaseloom_memory import PhaseLoomMemory
        
        mock_fields = MagicMock()
        mock_fields.alpha = np.array([1.0])
        mock_fields.beta = np.zeros((3, 3))
        mock_fields.K_sym6 = np.zeros((3, 3, 3, 3))
        
        memory = PhaseLoomMemory(mock_fields, base_dt=0.001)
        
        assert memory.step_count == 0
        assert memory.D_max_prev == 0.0
        assert memory.clock_system is None  # Not initialized yet
    
    def test_initialize_clock_system(self):
        """Test clock system initialization from PhaseLoomMemory."""
        from src.phaseloom.phaseloom_memory import PhaseLoomMemory
        
        mock_fields = MagicMock()
        mock_fields.alpha = np.array([1.0])
        mock_fields.beta = np.zeros((3, 3))
        mock_fields.K_sym6 = np.zeros((3, 3, 3, 3))
        
        mock_emitter = MagicMock()
        memory = PhaseLoomMemory(mock_fields, base_dt=0.001)
        
        clock_system = memory.initialize_clock_system(mock_emitter)
        
        assert clock_system is not None
        assert memory.clock_system is clock_system
    
    def test_should_compute_loom_with_clock_system(self):
        """Test clock-based triggering for loom computation."""
        from src.phaseloom.phaseloom_memory import PhaseLoomMemory
        
        mock_fields = MagicMock()
        mock_fields.alpha = np.array([1.0])
        mock_fields.beta = np.zeros((3, 3))
        mock_fields.K_sym6 = np.zeros((3, 3, 3, 3))
        
        mock_emitter = MagicMock()
        memory = PhaseLoomMemory(mock_fields, base_dt=0.001)
        memory.initialize_clock_system(mock_emitter)
        
        K = np.zeros((3, 3, 3, 3))
        gamma = np.zeros((3, 3, 3))
        
        # Step 1: Only band 0 is due (interval 1)
        should_compute, bands = memory.should_compute_loom(
            step=1, K=K, gamma=gamma, residual_slope=0.0,
            rollback_occurred=False, amplitude=1.0, dominant_band=0, resolution=64
        )
        
        assert should_compute
        # With 8 octaves, only band 0 is due at step 1 (interval=1)
        # FIX: Changed from 27 to 8 to match DEFAULT_OCTAVES in gr_clocks.py
        assert len(bands) == 8
        assert bands[0] == True  # Band 0 always updates at step 1
    
    def test_get_bands_to_update(self):
        """Test retrieval of bands to update."""
        from src.phaseloom.phaseloom_memory import PhaseLoomMemory
        
        mock_fields = MagicMock()
        mock_fields.alpha = np.array([1.0])
        mock_fields.beta = np.zeros((3, 3))
        mock_fields.K_sym6 = np.zeros((3, 3, 3, 3))
        
        mock_emitter = MagicMock()
        memory = PhaseLoomMemory(mock_fields, base_dt=0.001)
        memory.initialize_clock_system(mock_emitter)
        
        K = np.zeros((3, 3, 3, 3))
        gamma = np.zeros((3, 3, 3))
        
        # Step 8: Bands 0-3 should be due (intervals 1, 2, 4, 8)
        # Use dominant_band=3 so all lower bands (0-3) pass the culling check
        memory.should_compute_loom(
            step=8, K=K, gamma=gamma, residual_slope=0.0,
            rollback_occurred=False, amplitude=1.0, dominant_band=3, resolution=64
        )
        
        bands = memory.get_bands_to_update()
        # At step 8 with dominant_band=3, bands 0-3 should be updated
        for o in range(4):
            assert bands[o] == True, f"Band {o} should be updated at step 8"
    
    def test_get_clock_summary(self):
        """Test clock summary retrieval."""
        from src.phaseloom.phaseloom_memory import PhaseLoomMemory
        
        mock_fields = MagicMock()
        mock_fields.alpha = np.array([1.0])
        mock_fields.beta = np.zeros((3, 3))
        mock_fields.K_sym6 = np.zeros((3, 3, 3, 3))
        
        mock_emitter = MagicMock()
        memory = PhaseLoomMemory(mock_fields, base_dt=0.001)
        memory.initialize_clock_system(mock_emitter)
        
        summary = memory.get_clock_summary()
        
        assert 'global_step' in summary
        assert 'regime_hash' in summary
        assert 'culled_bands' in summary
    
    def test_post_loom_update(self):
        """Test post-loom update state management."""
        from src.phaseloom.phaseloom_memory import PhaseLoomMemory
        
        mock_fields = MagicMock()
        mock_fields.alpha = np.array([1.0])
        mock_fields.beta = np.zeros((3, 3))
        mock_fields.K_sym6 = np.zeros((3, 3, 3, 3))
        
        mock_emitter = MagicMock()
        memory = PhaseLoomMemory(mock_fields, base_dt=0.001)
        memory.initialize_clock_system(mock_emitter)
        
        loom_data = {
            'D_max': 0.5,
            'K': np.zeros((3, 3, 3, 3)),
            'gamma': np.zeros((3, 3, 3))
        }
        
        memory.post_loom_update(loom_data, step=5)
        
        assert memory.D_max_prev == 0.5
        assert memory.last_loom_step == 5
        assert memory.prev_K is not None


class TestLegacyCompatibility:
    """Tests for backward compatibility with legacy logic."""
    
    def test_legacy_fallback(self):
        """Test that legacy logic works when no clock system is set."""
        from src.phaseloom.phaseloom_memory import PhaseLoomMemory
        
        mock_fields = MagicMock()
        mock_fields.alpha = np.array([1.0])
        mock_fields.beta = np.zeros((3, 3))
        mock_fields.K_sym6 = np.zeros((3, 3, 3, 3))
        
        memory = PhaseLoomMemory(mock_fields, base_dt=0.001)
        # No clock system set
        
        K = np.zeros((3, 3, 3, 3))
        gamma = np.zeros((3, 3, 3))
        
        # Legacy logic: step < 4, no deltas, should return False
        should_compute, bands = memory.should_compute_loom(
            step=1, K=K, gamma=gamma, residual_slope=0.0,
            rollback_occurred=False, amplitude=1.0, dominant_band=0, resolution=64
        )
        
        # Legacy logic triggers at step >= 4
        assert not should_compute


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
