import unittest
import tempfile
import os
import json
import subprocess
import noetica_nsc_phase1.nsc_module as nsc_module

class TestBundleBitStable(unittest.TestCase):
    def test_bundle_bit_stable(self):
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
            
            # Build first
            bundle1 = os.path.join(tmpdir, 'bundle1.nscb')
            result1 = subprocess.run(
                ['python', '-m', 'noetica_nsc_phase1.nsc_cli', 'build-module', tmpdir, '--out', bundle1],
                capture_output=True, text=True, cwd='.'
            )
            self.assertEqual(result1.returncode, 0)
            
            # Build second
            bundle2 = os.path.join(tmpdir, 'bundle2.nscb')
            result2 = subprocess.run(
                ['python', '-m', 'noetica_nsc_phase1.nsc_cli', 'build-module', tmpdir, '--out', bundle2],
                capture_output=True, text=True, cwd='.'
            )
            self.assertEqual(result2.returncode, 0)
            
            # Read bytes
            with open(bundle1, 'rb') as f:
                bytes1 = f.read()
            with open(bundle2, 'rb') as f:
                bytes2 = f.read()
            self.assertEqual(bytes1, bytes2)