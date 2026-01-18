import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
from noetica_nsc_phase1 import nsc_cli

class TestCacheHitSkipsRebuild(unittest.TestCase):
    def test_cache_hit_skips_rebuild(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nsc_file = os.path.join(tmpdir, 'test.nsc')
            with open(nsc_file, 'w') as f:
                f.write('= ∇² u')
            bundle_out1 = os.path.join(tmpdir, 'out1.zip')
            bundle_out2 = os.path.join(tmpdir, 'out2.zip')

            counter = {'count': 0}

            def mock_nsc_to_pde(*args, **kwargs):
                counter['count'] += 1
                # Return dummy values: prog, flattened, bc, tpl
                prog = MagicMock()
                flattened = []
                bc = MagicMock()
                bc.opcodes = []
                bc.trace = []
                tpl = MagicMock()
                tpl.as_latex = MagicMock(return_value='')
                tpl.boundary = 'none'
                return prog, flattened, bc, tpl

            with patch('noetica_nsc_phase1.nsc.nsc_to_pde', side_effect=mock_nsc_to_pde):
                # First call - should compile
                nsc_cli.do_bundle(tmpdir, bundle_out1, cache=True)
                first_count = counter['count']
                self.assertGreater(first_count, 0, "First call should have compiled")

                # Second call - should hit cache, no compile
                nsc_cli.do_bundle(tmpdir, bundle_out2, cache=True)
                second_count = counter['count']
                self.assertEqual(second_count, first_count, "Compile should not be called on cache hit")

if __name__ == '__main__':
    unittest.main()