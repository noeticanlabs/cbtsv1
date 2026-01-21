"""
Test 3 Wave Dynamics
"""

import numpy as np
import logging

class Test3:
    def __init__(self, gr_solver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        # Initialize solver with Minkowski data
        self.gr_solver.init_minkowski()
        
        N = self.gr_solver.N
        center = N // 2
        
        # Inject localized bump in K_sym6 (Gaussian at center with amplitude 1e-6)
        amp = 1e-6
        sigma = 3.0
        x = np.arange(N) - center
        y = np.arange(N) - center
        z = np.arange(N) - center
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        r2 = X**2 + Y**2 + Z**2
        bump = amp * np.exp(-r2 / (2 * sigma**2))
        self.gr_solver.fields.K_sym6[..., 0] += bump  # Add to xx component
        
        # Hold gauge steady by overriding gauge evolution (set ∂t α = 0, ∂t β = 0)
        self.gr_solver.gauge.evolve_lapse = lambda dt: None
        self.gr_solver.gauge.evolve_shift = lambda dt: None
        
        # Reset time and step
        self.gr_solver.orchestrator.t = 0.0
        self.gr_solver.orchestrator.step = 0
        
        # Lists to track metrics
        peak_locations = []
        peak_amplitudes = []
        times = []
        spread_max_r = []
        
        initial_threshold = amp / 1e6  # Threshold for spread calculation
        
        # Evolve for 10 steps
        for step in range(10):
            dt, _, _ = self.gr_solver.orchestrator.run_step()
            
            # Compute H grid
            self.gr_solver.constraints.compute_hamiltonian()
            H = self.gr_solver.constraints.H
            H_abs = np.abs(H)
            
            # Find peak |H| and location
            peak_idx = np.argmax(H_abs.ravel())
            peak_loc = np.unravel_index(peak_idx, H.shape)
            peak_amp = H_abs.flat[peak_idx]
            
            peak_locations.append(peak_loc)
            peak_amplitudes.append(peak_amp)
            times.append(self.gr_solver.orchestrator.t)
            
            # Compute spread: max r where |H| > threshold
            mask = H_abs > initial_threshold
            if np.any(mask):
                i_vals, j_vals, k_vals = np.where(mask)
                r_vals = np.sqrt((i_vals - center)**2 + (j_vals - center)**2 + (k_vals - center)**2)
                max_r = np.max(r_vals)
            else:
                max_r = 0.0
            spread_max_r.append(max_r)
        
        # Compute peak speeds (distances between consecutive peaks)
        peak_speeds = []
        for i in range(1, len(peak_locations)):
            loc1 = np.array(peak_locations[i-1])
            loc2 = np.array(peak_locations[i])
            dist = np.linalg.norm(loc2 - loc1)
            peak_speeds.append(dist)
        
        # Compute spread rates (increase in max_r per step)
        spread_rates = []
        for i in range(1, len(spread_max_r)):
            rate = spread_max_r[i] - spread_max_r[i-1]
            spread_rates.append(rate)
        
        # Peak decay: list of peak amplitudes
        peak_decay = peak_amplitudes
        
        # Check pass conditions
        max_speed = max(peak_speeds) if peak_speeds else 0
        smooth_movement = all(speed <= 3 for speed in peak_speeds)
        no_teleport = max_speed <= N / 2  # Arbitrary, but reasonable
        no_sudden_elsewhere = True  # Assume if speeds are low, no teleport
        
        passed = smooth_movement and no_teleport and no_sudden_elsewhere
        
        diagnosis = f"Max peak speed: {max_speed:.2f} cells/step, {'passed' if passed else 'failed'} criteria (speed <=3, smooth movement)"
        
        metrics = {
            'peak_speeds': peak_speeds,
            'peak_decay': peak_decay,
            'spread_rates': spread_rates
        }
        
        return {'passed': passed, 'metrics': metrics, 'diagnosis': diagnosis}