import unittest

import noetica_nsc_phase1.nsc as nsc
import noetica_nsc_phase1.nsc_diag as nsc_diag

class TestTokenize(unittest.TestCase):
    def test_strip_whitespace(self):
        result = nsc.tokenize(" a + b ")
        self.assertEqual(result, ['a', '+', 'b'])

    def test_empty_input_error(self):
        with self.assertRaises(nsc_diag.NSCError) as cm:
            nsc.tokenize("")
        self.assertEqual(cm.exception.code, nsc_diag.E_EMPTY_INPUT)