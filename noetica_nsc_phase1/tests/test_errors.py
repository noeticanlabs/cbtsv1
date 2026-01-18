import unittest

import noetica_nsc_phase1.nsc as nsc
import noetica_nsc_phase1.nsc_diag as nsc_diag

class TestErrors(unittest.TestCase):
    def test_parse_unclosed_group(self):
        # Unclosed group
        with self.assertRaises(nsc_diag.NSCError) as cm:
            nsc.nsc_to_pde("[u")
        self.assertEqual(cm.exception.code, 8)  # E_PARSE_UNCLOSED_GROUP

    def test_parse_unexpected_token(self):
        # Unexpected token, e.g., ) without (
        with self.assertRaises(nsc_diag.NSCError) as cm:
            nsc.nsc_to_pde("u)")
        self.assertEqual(cm.exception.code, 7)  # E_PARSE_UNEXPECTED_TOKEN

    def test_noncanonical_unicode(self):
        # Non-canonical Unicode raises E_NONCANONICAL_UNICODE
        with self.assertRaises(nsc_diag.NSCError) as cm:
            nsc.nsc_to_pde("u\u0301")  # e with combining acute
        self.assertEqual(cm.exception.code, 3)  # E_NONCANONICAL_UNICODE

    def test_empty_input(self):
        # Empty input raises E_EMPTY_INPUT
        with self.assertRaises(nsc_diag.NSCError) as cm:
            nsc.nsc_to_pde("")
        self.assertEqual(cm.exception.code, 1)  # E_EMPTY_INPUT

