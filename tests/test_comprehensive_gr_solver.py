#!/usr/bin/env python3
"""
Comprehensive GR Solver System Test
Verifies integration of Physics, PhaseLoom, Rails, and Memory.
"""

import sys
import os
import numpy as np
import unittest
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gr_solver.gr_solver import GRSolver
from phaseloom_27 import PhaseLoom27

class TestComprehensiveGRSolver(unittest.TestCase):
    def setUp(self):
        # Use small grid for speed, with values from environment variables
        self.Nx = int(os.environ.get('N', 16))
        self.Ny = int(os.environ.get('N', 16))
        self.Nz = int(os.environ.get('N', 16))
        self.dt = float(os.environ.get('DT', 0.1))
        
        self.solver = GRSolver(Nx=self.Nx, Ny=self.Ny, Nz=self.Nz, dx=self.dt, dy=self.dt, dz=self.dt, log_level=logging.ERROR)
        self.solver.init_minkowski()

    def test_phaseloom_rails_logic(self):
        """Verify the new PhaseLoom rails logic directly."""
        pl = PhaseLoom27()
        
        # Test check_gate_orch (Section 6 implementation)
        window_stats = {'chatter_score': 0.6, 'max_residual': 1e-5}
        thresholds = {'chatter': 0.5, 'residual': 1e-4}
        
        # Should fail due to chatter
        passed, reasons = pl.check_gate_orch(window_stats, thresholds)
        self.assertFalse(passed)
        self.assertTrue(any("Chatter" in r for r in reasons))
        
        # Should pass with lower chatter
        window_stats['chatter_score'] = 0.4
        passed, reasons = pl.check_gate_orch(window_stats, thresholds)
        self.assertTrue(passed)

        # Test get_rails (Section 6 implementation)
        # PHY.H -> increase dissipation
        rails = pl.get_rails(('PHY', 'H', 'R1')) 
        self.assertTrue(len(rails) > 0)
        self.assertEqual(rails[0]['action'], 'increase_dissipation')

        # SEM.R2 -> halt
        rails = pl.get_rails(('SEM', 'L', 'R2'))
        self.assertTrue(len(rails) > 0)
        self.assertEqual(rails[0]['action'], 'halt_and_dump')

    def test_solver_integration(self):
        """Verify solver runs and integrates PhaseLoom."""
        # Inject noise to trigger dynamics
        self.solver.fields.gamma_sym6[..., 0] += np.random.normal(0, 1e-5, (self.Nx, self.Ny, self.Nz))
        
        # Run a few steps
        self.solver.run(T_max=0.05, dt_max=self.dt)
            
        # Check if time advanced
        self.assertGreater(self.solver.t, 0.0)
        self.assertGreater(self.solver.step, 0)

if __name__ == '__main__':
    unittest.main()
