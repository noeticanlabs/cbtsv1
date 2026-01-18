import unittest

import noetica_nsc_phase1.nsc as nsc

class TestPDEAssemble(unittest.TestCase):
    def test_example_01_coefficients(self):
        # Test assemble with nsc_to_pde for "= ∇² u"
        prog, flat, bc, tpl = nsc.nsc_to_pde("= ∇² u")
        self.assertEqual(tpl.terms, {'\\nabla^2 u': 1.0})