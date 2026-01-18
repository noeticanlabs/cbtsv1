import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
from noetica_nsc_phase1 import nsc_cli, nsc_cache

class TestCacheValidationRejectsTamper(unittest.TestCase):
    def test_cache_validation_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nsc_file = os.path.join(tmpdir, 'test.nsc')
            with open(nsc_file, 'w') as f:
                f.write('= ∇² u')
            bundle_out1 = os.path.join(tmpdir, 'out1.zip')
            bundle_out2 = os.path.join(tmpdir, 'out2.zip')

            counter = {'count': 0}

            def mock_nsc_to_pde(*args, **kwargs):
                counter['count'] += 1
                # Return dummy values
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
                # First call - should compile and cache
                nsc_cli.do_bundle(tmpdir, bundle_out1, cache=True)
                first_count = counter['count']
                self.assertGreater(first_count, 0, "First call should have compiled")

                # Tamper with cached bytecode.bin
                # First, find the cache dir
                # Need to get module_id from the bundle
                import zipfile
                import json
                with zipfile.ZipFile(bundle_out1, 'r') as zf:
                    with zf.open('manifest.json') as f:
                        manifest = json.loads(f.read().decode('utf-8'))
                module_id = manifest['module_id']
                cache_d = nsc_cache.cache_dir(module_id)
                bytecode_path = os.path.join(cache_d, 'bytecode.bin')
                if os.path.exists(bytecode_path):
                    # Tamper by appending junk
                    with open(bytecode_path, 'ab') as f:
                        f.write(b'tamper')

                # Second call - should detect tamper and rebuild
                nsc_cli.do_bundle(tmpdir, bundle_out2, cache=True)
                second_count = counter['count']
                self.assertGreater(second_count, first_count, "Compile should be called after tamper detection")

if __name__ == '__main__':
    unittest.main()