import unittest

import noetica_nsc_phase1.nsc as nsc

class TestCompileBytecodeFromAST(unittest.TestCase):
    def test_bytecode_assertion(self):
        src = "[φ↻]⇒[∆◯]□"
        prog, flat, bc, tpl = nsc.nsc_to_pde(src)
        expected_opcodes = [1, 2, 7, 6, 5, 8]  # φ, ↻, ⇒, ∆, ◯, □
        self.assertEqual(bc.opcodes, expected_opcodes)