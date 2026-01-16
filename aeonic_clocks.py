import hashlib
import numpy as np

class AeonicClockPack:
    def __init__(self):
        self.tau_s = 0  # stepper time
        self.tau_l = 0  # loom time
        self.tau_m = 0  # maintenance time

        # Regime hashes
        self.stepper_regime_hash = None
        self.loom_regime_hash = None

        # Tracking for stepper regime
        self.dt_bucket = 0
        self.rollback_rate = 0.0
        self.residual_slope = 0.0
        self.rollback_count = 0
        self.attempt_count = 0

        # Tracking for loom regime
        self.D_state = 0.0
        self.dominant_band_index = 0
        self.spectral_centroid = 0.0
        self.transfer_pattern = 0

    def tick_step_attempt(self, dt, accepted: bool, rolled_back: bool):
        self.tau_s += 1
        self.attempt_count += 1
        if rolled_back:
            self.rollback_count += 1
        # Update dt bucket (log scale)
        self.dt_bucket = int(np.log10(dt) * 10) if dt > 0 else -100
        # Update rollback rate (EWMA)
        alpha = 0.1
        self.rollback_rate = alpha * (1 if rolled_back else 0) + (1 - alpha) * self.rollback_rate

    def tick_loom_update(self):
        self.tau_l += 1

    def tick_maintenance(self):
        self.tau_m += 1

    def update_stepper_regime(self, residual_current, residual_prev, rollback_rate):
        # Residual slope (if prev available)
        if residual_prev is not None:
            self.residual_slope = (residual_current - residual_prev) / max(residual_current, 1e-15)
        # Update hash
        hash_input = f"{self.dt_bucket}:{int(self.rollback_rate*100)}:{int(self.residual_slope*1000)}"
        self.stepper_regime_hash = hashlib.md5(hash_input.encode()).hexdigest()[:16]

    def update_loom_regime(self, D_state, dominant_band_index, spectral_centroid, transfer_pattern):
        self.D_state = D_state
        self.dominant_band_index = dominant_band_index
        self.spectral_centroid = spectral_centroid
        self.transfer_pattern = transfer_pattern
        hash_input = f"{int(D_state*1e6)}:{dominant_band_index}:{int(spectral_centroid*100)}:{transfer_pattern}"
        self.loom_regime_hash = hashlib.md5(hash_input.encode()).hexdigest()[:16]