import unittest

import noetica_nsc_phase1.nsc as nsc


class TestTracePathsAreDeterministic(unittest.TestCase):
    def test_trace_paths_are_deterministic(self):
        input_str = "[φ↻]⇒[∆◯]□"
        prog1, flat1, bc1, pde1 = nsc.nsc_to_pde(input_str)
        prog2, flat2, bc2, pde2 = nsc.nsc_to_pde(input_str)
        paths1 = [te.path for te in bc1.trace]
        paths2 = [te.path for te in bc2.trace]
        self.assertEqual(paths1, paths2)