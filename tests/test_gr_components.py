import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gr_solver.gr_core_fields import GRCoreFields
from gr_solver.gr_geometry import GRGeometry
from gr_solver.gr_constraints import GRConstraints
from gr_solver.gr_rhs import GRRhs
from gr_solver.gr_ledger import GRLedger
from gr_solver.gr_scheduler import GRScheduler
from gr_solver.gr_gates import GateChecker
from gr_solver.gr_loc import GRLoC

class TestGRComponents(unittest.TestCase):
    def setUp(self):
        self.Nx, self.Ny, self.Nz = 16, 16, 16
        self.fields = GRCoreFields(self.Nx, self.Ny, self.Nz)
        self.fields.init_minkowski()
        self.geometry = GRGeometry(self.fields)
        self.constraints = GRConstraints(self.fields, self.geometry)
        self.loc_operator = GRLoC(self.fields, self.geometry, self.constraints)

    def test_rhs_component(self):
        """Test GRRhs component initialization and computation."""
        rhs = GRRhs(self.fields, self.geometry, self.constraints, self.loc_operator)
        rhs.compute_rhs(t=0.0)
        self.assertIsNotNone(rhs.rhs_gamma_sym6)
        self.assertEqual(rhs.rhs_gamma_sym6.shape, (self.Nx, self.Ny, self.Nz, 6))
        # Check that RHS is zero for Minkowski (approximately)
        self.assertTrue(np.all(np.abs(rhs.rhs_gamma_sym6) < 1e-10))

    def test_ledger_component(self):
        """Test GRLedger component receipt emission."""
        ledger = GRLedger(receipts_file="test_receipts.jsonl")
        ledger.emit_clock_decision_receipt(0, 0.0, 0.1, self.fields, {'dt_CFL': 0.1})
        self.assertTrue(len(ledger.receipts) > 0)
        self.assertEqual(ledger.receipts[-1]['event'], 'CLOCK_DECISION')

    def test_scheduler_component(self):
        """Test GRScheduler component clock computation."""
        scheduler = GRScheduler(self.fields)
        clocks, dt = scheduler.compute_clocks(0.1)
        self.assertIn('dt_CFL', clocks)
        self.assertTrue(dt <= 0.1)

    def test_gatekeeper_component(self):
        """Test GRGatekeeper component gate checking."""
        gatekeeper = GateChecker(self.constraints) # Updated to GateChecker
        # Mock constraints
        self.constraints.eps_H = 0.0
        self.constraints.eps_M = 0.0
        self.constraints.eps_proj = 0.0
        self.constraints.eps_clk = 0.0
        
        accepted, hard_fail, penalty, reasons, margins, corrections = gatekeeper.check_gates() # Updated method call
        self.assertTrue(accepted)
        self.assertFalse(hard_fail)
        self.assertEqual(len(reasons), 0)

if __name__ == '__main__':
    unittest.main()