import unittest
import subprocess
import os
from noetica_nsc_phase1.nsc import tokenize

class TestExplainOpSpanMatchesTokens(unittest.TestCase):
    def test_explain_op_span_matches_tokens(self):
        src_file = 'noetica_nsc_phase1/examples/example_01.nsc'
        # Find the index of ↻
        with open(src_file, 'r') as f:
            source = f.read()
        tokens = tokenize(source)
        arrow_index = tokens.index('↻')

        result = subprocess.run(
            ['python', '-m', 'noetica_nsc_phase1.nsc_cli', 'explain-op', '--src', src_file, '--index', str(arrow_index)],
            capture_output=True, text=True, cwd='.'
        )
        self.assertEqual(result.returncode, 0)
        output = result.stdout
        # Parse span_start and span_end
        lines = output.strip().split('\n')
        span_start = None
        span_end = None
        for line in lines:
            if line.startswith('span_start:'):
                span_start = int(line.split(':')[1].strip())
            elif line.startswith('span_end:'):
                span_end = int(line.split(':')[1].strip())
        self.assertIsNotNone(span_start)
        self.assertIsNotNone(span_end)
        # Check that tokens[span_start:span_end] == ['↻']
        token_slice = tokens[span_start:span_end]
        self.assertEqual(token_slice, ['↻'])

if __name__ == '__main__':
    unittest.main()