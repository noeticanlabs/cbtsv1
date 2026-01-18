import unittest
import tempfile
import os
import json
import noetica_nsc_phase1.nsc_module as nsc_module
import noetica_nsc_phase1.nsc_diag as nsc_diag

class TestSourcesMustBeSorted(unittest.TestCase):
    def test_sources_must_be_sorted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, 'nsc.module.json')
            manifest = {
                'sources': ['b.nsc', 'a.nsc'],  # out of order
                'imports': []
            }
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f)
            with self.assertRaises(nsc_diag.NSCError) as cm:
                nsc_module.load_module_manifest(tmpdir)
            self.assertEqual(cm.exception.code, nsc_diag.E_MANIFEST_SCHEMA)