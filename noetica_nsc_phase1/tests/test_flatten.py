import unittest

import noetica_nsc_phase1.nsc as nsc

class TestFlatten(unittest.TestCase):
    def test_flattened_list(self):
        src = "[φ↻]⇒[∆◯]□"
        prog, flat, bc, tpl = nsc.nsc_to_pde(src)
        expected_flat = ['φ', '↻', '⇒', '∆', '◯', '□']
        self.assertEqual(flat, expected_flat)