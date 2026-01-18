import unittest

import noetica_nsc_phase1.nsc as nsc
import noetica_nsc_phase1.nsc_diag as nsc_diag

class TestBytecode(unittest.TestCase):
    def test_map_known_glyphs(self):
        tokens = ['+', '-', 'u', 'v']
        bc = nsc.compile_to_bytecode(tokens)
        self.assertEqual(bc.opcodes, [9, 10, 'u', 'v'])

    def test_unknown_error(self):
        tokens = ['@']
        with self.assertRaises(nsc_diag.NSCError) as cm:
            nsc.compile_to_bytecode(tokens)
        self.assertEqual(cm.exception.code, nsc_diag.E_UNKNOWN_GLYPH)