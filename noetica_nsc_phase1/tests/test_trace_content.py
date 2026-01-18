import unittest

import noetica_nsc_phase1.nsc as nsc


class TestTraceContent(unittest.TestCase):
    def test_trace_content(self):
        prog, flat_glyphs, bc, pde = nsc.nsc_to_pde("[φ↻]⇒[∆◯]□")
        self.assertEqual(flat_glyphs, ['φ', '↻', '⇒', '∆', '◯', '□'])
        self.assertEqual(bc.opcodes, [1, 2, 7, 6, 5, 8])