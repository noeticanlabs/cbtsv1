import unittest

import noetica_nsc_phase1.nsc as nsc


class TestArrowSpan(unittest.TestCase):
    def test_arrow_span(self):
        input_str = "[φ↻]⇒[∆◯]□"
        prog, flat_glyphs, bc, pde = nsc.nsc_to_pde(input_str)
        arrow_idx = input_str.index('⇒')
        self.assertEqual(bc.trace[2].span.start, arrow_idx)
        self.assertEqual(bc.trace[2].span.end, arrow_idx + 1)