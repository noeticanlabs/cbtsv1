#!/usr/bin/env python3
"""
Test Gate Step Logic
Verifies that PhaseLoom27 enforces the LoC-GR default thresholds correctly.
"""

import unittest
import sys
import os

# Add project root to path to import phaseloom_27
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phaseloom_27 import PhaseLoom27

class TestGateStepLogic(unittest.TestCase):
    def setUp(self):
        self.loom = PhaseLoom27()
        # Reset all residuals to 0 (default state)
        for d in self.loom.DOMAINS:
            for s in self.loom.SCALES:
                self.loom.update_residual(d, s, 0.0)

    def test_clean_state_passes(self):
        """Ensure a clean state with 0 residuals passes."""
        passed, reasons = self.loom.check_gate_step()
        self.assertTrue(passed, f"Clean state failed: {reasons}")
        self.assertEqual(len(reasons), 0)

    def test_phy_thresholds(self):
        """Verify PHY threshold (1e-4)."""
        # Just below threshold -> Pass
        self.loom.update_residual('PHY', 'L', 0.9e-4)
        passed, reasons = self.loom.check_gate_step()
        self.assertTrue(passed, f"PHY=0.9e-4 should pass. Reasons: {reasons}")

        # Just above threshold -> Fail
        self.loom.update_residual('PHY', 'L', 1.1e-4)
        passed, reasons = self.loom.check_gate_step()
        self.assertFalse(passed, "PHY=1.1e-4 should fail")
        self.assertTrue(any("PHY Barrier violation" in r for r in reasons))

    def test_cons_thresholds(self):
        """Verify CONS threshold (1e-6)."""
        # Just below threshold -> Pass
        self.loom.update_residual('CONS', 'M', 0.9e-6)
        passed, reasons = self.loom.check_gate_step()
        self.assertTrue(passed, f"CONS=0.9e-6 should pass. Reasons: {reasons}")

        # Just above threshold -> Fail
        self.loom.update_residual('CONS', 'M', 1.1e-6)
        passed, reasons = self.loom.check_gate_step()
        self.assertFalse(passed, "CONS=1.1e-6 should fail")
        self.assertTrue(any("CONS Barrier violation" in r for r in reasons))

    def test_sem_thresholds(self):
        """Verify SEM threshold (0.0)."""
        # Zero -> Pass
        self.loom.update_residual('SEM', 'H', 0.0)
        passed, reasons = self.loom.check_gate_step()
        self.assertTrue(passed)

        # Any positive value -> Fail
        self.loom.update_residual('SEM', 'H', 1e-12)
        passed, reasons = self.loom.check_gate_step()
        self.assertFalse(passed, "SEM > 0 should fail")
        self.assertTrue(any("SEM Barrier violation" in r for r in reasons))

    def test_mixed_failures(self):
        """Verify multiple domain failures are reported."""
        self.loom.update_residual('PHY', 'L', 1.0)
        self.loom.update_residual('SEM', 'L', 1.0)
        
        passed, reasons = self.loom.check_gate_step()
        self.assertFalse(passed)
        self.assertTrue(len(reasons) >= 2, "Should report multiple reasons")

if __name__ == '__main__':
    unittest.main()