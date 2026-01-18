import unittest

import noetica_nsc_phase1.nsc as nsc


class TestTraceLengths(unittest.TestCase):
    def test_trace_lengths(self):
        prog, flat_glyphs, bc, pde = nsc.nsc_to_pde("[φ↻]⇒[∆◯]□")
        self.assertEqual(len(bc.opcodes), 6)
        self.assertEqual(len(bc.trace), 6)