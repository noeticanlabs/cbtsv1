import unittest
import tempfile
import os
import json
import zipfile
import subprocess

class TestImportVerificationFailsOnWrongModuleId(unittest.TestCase):
    def test_import_verification_fails_on_wrong_module_id(self):
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
            
            # Tamper the bundle: change module_id in manifest
            with zipfile.ZipFile(bundle, 'a') as zf:
                # Read current manifest
                with zf.open('manifest.json') as f:
                    data = json.loads(f.read().decode('utf-8'))
                data['module_id'] = 'wrong_id'
                # Write back
                zf.writestr('manifest.json', json.dumps(data))
            
            # Verify, expect fail
            result_verify = subprocess.run(
                ['python', '-m', 'noetica_nsc_phase1.nsc_cli', 'verify-module', bundle],
                capture_output=True, text=True, cwd='.'
            )
            self.assertEqual(result_verify.returncode, 0)  # CLI doesn't fail, but
            self.assertIn("verified=False", result_verify.stdout)