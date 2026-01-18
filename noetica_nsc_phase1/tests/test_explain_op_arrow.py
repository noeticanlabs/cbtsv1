import unittest
import subprocess
import os

class TestExplainOpArrow(unittest.TestCase):
    def test_explain_op_arrow(self):
        src_file = 'noetica_nsc_phase1/examples/example_01.nsc'
        result = subprocess.run(
            ['python', '-m', 'noetica_nsc_phase1.nsc_cli', 'explain-op', '--src', src_file, '--index', '2'],
            capture_output=True, text=True, cwd='.'
        )
        self.assertEqual(result.returncode, 0)
        output = result.stdout
        # Check that glyph is ↻ for index 2
        self.assertIn("glyph: ↻", output)
        # Perhaps check other fields
        self.assertIn("opcode:", output)
        self.assertIn("meaning:", output)
        self.assertIn("path:", output)
        self.assertIn("span_start:", output)
        self.assertIn("span_end:", output)

if __name__ == '__main__':
    unittest.main()