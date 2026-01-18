import unittest
import tempfile
import os
import json
import subprocess

class TestVerifyModuleRoundtrip(unittest.TestCase):
    def test_verify_module_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create manifest
            manifest_path = os.path.join(tmpdir, 'nsc.module.json')
            manifest = {
                'sources': ['a.nsc'],
                'imports': []
            }
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f)
            
            # Create src file
            with open(os.path.join(tmpdir, 'a.nsc'), 'w') as f:
                f.write('= ∇ u')
            
            # Build
            bundle = os.path.join(tmpdir, 'bundle.nscb')
            result_build = subprocess.run(
                ['python', '-m', 'noetica_nsc_phase1.nsc_cli', 'build-module', tmpdir, '--out', bundle],
                capture_output=True, text=True, cwd='.'
            )
            self.assertEqual(result_build.returncode, 0)
            
            # Verify
            result_verify = subprocess.run(
                ['python', '-m', 'noetica_nsc_phase1.nsc_cli', 'verify-module', bundle],
                capture_output=True, text=True, cwd='.'
            )
            self.assertEqual(result_verify.returncode, 0)
            self.assertIn("verified=True", result_verify.stdout)