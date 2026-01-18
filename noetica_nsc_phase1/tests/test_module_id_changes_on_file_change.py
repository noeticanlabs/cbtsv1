import unittest
import tempfile
import os
import json
import noetica_nsc_phase1.nsc_module as nsc_module

class TestModuleIdChangesOnFileChange(unittest.TestCase):
    def test_module_id_changes_when_one_file_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create manifest
            manifest_path = os.path.join(tmpdir, 'nsc.module.json')
            manifest = {
                'sources': ['a.nsc', 'b.nsc'],
                'imports': []
            }
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f)
            
            # Create src files
            with open(os.path.join(tmpdir, 'a.nsc'), 'w') as f:
                f.write('= ∇ u')
            with open(os.path.join(tmpdir, 'b.nsc'), 'w') as f:
                f.write('= ∇² u')
            
            # Build first
            loaded_manifest = nsc_module.load_module_manifest(tmpdir)
            artifact1 = nsc_module.compile_module(tmpdir, loaded_manifest)
            id1 = artifact1.module_id
            
            # Change one glyph in b.nsc
            with open(os.path.join(tmpdir, 'b.nsc'), 'w') as f:
                f.write('= ∇³ u')  # changed ∇² to ∇³
            
            # Build second
            artifact2 = nsc_module.compile_module(tmpdir, loaded_manifest)
            id2 = artifact2.module_id
            
            self.assertNotEqual(id1, id2)